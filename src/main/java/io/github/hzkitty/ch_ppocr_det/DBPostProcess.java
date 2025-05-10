package io.github.hzkitty.ch_ppocr_det;

import de.lighti.clipper.Clipper;
import de.lighti.clipper.ClipperOffset;
import de.lighti.clipper.Path;
import de.lighti.clipper.Paths;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.*;

/**
 * 用于对DB模型的输出做后处理，生成文本检测框
 */
public class DBPostProcess {

    // 二值化阈值
    private final float thresh;
    // 最低得分阈值
    public float boxThresh;
    // 最大候选框数量
    private final int maxCandidates;
    // 扩张比率
    public float unclipRatio;
    // 最小框尺寸，用于过滤小目标
    private final int minSize = 3;
    // 计算得分的模式: "fast" / "slow"
    private final String scoreMode;
    // 膨胀核
    private Mat dilationKernel = null;

    /**
     * 构造函数
     * @param thresh         二值化阈值
     * @param boxThresh      最低得分阈值
     * @param maxCandidates  最大候选框数
     * @param unclipRatio    扩张比率
     * @param scoreMode      "fast" or "slow"
     * @param useDilation    是否使用膨胀操作
     */
    public DBPostProcess(float thresh, float boxThresh, int maxCandidates, float unclipRatio, String scoreMode, boolean useDilation) {
        this.thresh = thresh;
        this.boxThresh = boxThresh;
        this.maxCandidates = maxCandidates;
        this.unclipRatio = unclipRatio;
        this.scoreMode = scoreMode != null ? scoreMode : "fast";

        if (useDilation) {
            Mat kernel = Mat.ones(new Size(2, 2), CvType.CV_8UC1);
            this.dilationKernel = kernel;
        }
    }

    /**
     * 后处理主入口，对预测输出做阈值化、膨胀、找轮廓、生成检测框等
     *
     * @param pred      模型输出预测，形状 [N, 1, H, W]，此处只处理 N=1 的情况
     * @param oriHeight 原始图像高度
     * @param oriWidth  原始图像宽度
     * @return 点位信息和得分
     */
    public ResultBundle call(float[][][][] pred, int oriHeight, int oriWidth) {
        // pred 的形状 [N, 1, H, W]，这里只处理单张图 pred[0][0][...]
        float[][] probMap = pred[0][0];  // shape [H][W]

        int h = probMap.length;       // H
        int w = probMap[0].length;    // W

        // 1. 生成二值化掩码 segmentation = pred > thresh
        byte[] maskData = new byte[h * w];
        int idx = 0;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                float val = probMap[row][col];
                maskData[idx++] = (byte) ((val > this.thresh) ? 255 : 0);
            }
        }
        // 一次性填充到 OpenCV 的 Mat 中，避免多次 JNI 调用
        Mat mask = new Mat(h, w, CvType.CV_8UC1);
        mask.put(0, 0, maskData);

        // 2. 若使用了膨胀操作，则在mask上执行dilate
        if (this.dilationKernel != null) {
            Imgproc.dilate(mask, mask, this.dilationKernel);
        }

        // 3. 从mask提取文本框(轮廓) + 计算score
        return this.boxesFromBitmap(probMap, mask, oriWidth, oriHeight);
    }

    /**
     * 从二值化掩码中提取文本框 + 计算分数
     *
     * @param pred       概率图 (H, W)
     * @param bitmap     二值化掩码 (H, W)
     * @param destWidth  原图宽度
     * @param destHeight 原图高度
     * @return (boxes, scores) => List<BoxPoints> + List<Float>
     */
    private ResultBundle boxesFromBitmap(float[][] pred, Mat bitmap, int destWidth, int destHeight) {
        int h = bitmap.rows();
        int w = bitmap.cols();

        // 1. findContours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        // 这里相当于 Python: cv2.findContours((bitmap*255).astype(np.uint8), ...)
        // 但我们已经在上面把bitmap构建成0/255，所以直接传bitmap
        Imgproc.findContours(bitmap, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // 限制最大候选框数量
        int numContours = Math.min(contours.size(), this.maxCandidates);

        List<Point[]> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        // 2. 遍历轮廓
        for (int i = 0; i < numContours; i++) {
            Point[] contour = contours.get(i).toArray();
            // 把四个点按 x 坐标排序并处理 y 坐标顺序，得到 box
            BoxAndSize boxAndSize = getMiniBoxes(new MatOfPoint2f(contour));
            Point[] box = boxAndSize.box;
            float sside = boxAndSize.minSideLen;
            if (sside < this.minSize) {
                continue;
            }

            // 2.3 计算score
            float score;
            if ("fast".equalsIgnoreCase(this.scoreMode)) {
                score = boxScoreFast(pred, box);
            } else {
                score = boxScoreSlow(pred, contour);
            }
            if (score < this.boxThresh) {
                continue;
            }

            // 2.4 unclip扩展
            Point[] expanded = unclip(box);
            // 再获取新的最小外接矩形
            BoxAndSize boxAndSize2 = getMiniBoxes(new MatOfPoint2f(expanded));
            Point[] newBox = boxAndSize2.box;
            float sside2 = boxAndSize2.minSideLen;
            if (sside2 < this.minSize + 2) {
                continue;
            }

            // 2.5 映射回原图
            Point[] mapped = mapToOriginalSize(newBox, w, h, destWidth, destHeight);
            boxes.add(mapped);
            scores.add(score);
        }

        return new ResultBundle(boxes, scores);
    }

    /**
     * 将四个点 newBox 映射回原图大小
     */
    private Point[] mapToOriginalSize(Point[] newBox, int curW, int curH, int oriW, int oriH) {
        // newBox[i].x, newBox[i].y 在 [0, curW], [0, curH] 范围
        // 要映射到 [0, oriW], [0, oriH]
        Point[] mapped = new Point[4];
        for (int i = 0; i < 4; i++) {
            double x = Math.round((newBox[i].x / (double) curW) * oriW);
            double y = Math.round((newBox[i].y / (double) curH) * oriH);
            // clip到范围
            double clippedX = Math.max(0, Math.min(x, oriW));
            double clippedY = Math.max(0, Math.min(y, oriH));
            mapped[i] = new Point(clippedX, clippedY);
        }
        return mapped;
    }

    public static BoxAndSize getMiniBoxes(MatOfPoint2f point2f) {
        MatOfPoint2f contour = new MatOfPoint2f(point2f);  // 将 Point[] 转换为 MatOfPoint2f
        RotatedRect boundingBox = Imgproc.minAreaRect(contour);  // 计算最小外接矩形

        Point[] points = new Point[4];  // 初始化一个点数组
        boundingBox.points(points);  // 通过 minAreaRect 获取四个顶点

        Arrays.sort(points, Comparator.comparingDouble(p -> p.x));  // 按 x 坐标排序点

        // 确定四个顶点的顺序
        int index1, index2, index3, index4;
        if (points[1].y > points[0].y) {
            index1 = 0;
            index4 = 1;
        } else {
            index1 = 1;
            index4 = 0;
        }

        if (points[3].y > points[2].y) {
            index2 = 2;
            index3 = 3;
        } else {
            index2 = 3;
            index3 = 2;
        }

        Point[] box = new Point[]{
                points[index1], points[index2], points[index3], points[index4]
        };

        float minSide = (float) Math.min(boundingBox.size.width, boundingBox.size.height);  // 计算最小边长
        return new BoxAndSize(box, minSide);
    }

    /**
     * 计算四边形四条边里最短的一条边长
     */
    private float getMinSideOfQuad(Point[] box) {
        // box依次 [p0, p1, p2, p3]
        // 边 p0->p1, p1->p2, p2->p3, p3->p0
        double d1 = dist(box[0], box[1]);
        double d2 = dist(box[1], box[2]);
        double d3 = dist(box[2], box[3]);
        double d4 = dist(box[3], box[0]);
        return (float) Math.min(Math.min(d1, d2), Math.min(d3, d4));
    }

    private double dist(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    /**
     * 计算文本框平均得分(FAST)，对应 Python box_score_fast
     *
     * @param bitmap 概率图 (H, W)
     * @param box    4个点
     * @return 平均得分
     */
    private float boxScoreFast(float[][] bitmap, Point[] box) {
        int h = bitmap.length;
        int w = bitmap[0].length;

        double xmin = Double.MAX_VALUE, xmax = -Double.MAX_VALUE;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE;
        for (Point p : box) {
            if (p.x < xmin) xmin = p.x;
            if (p.x > xmax) xmax = p.x;
            if (p.y < ymin) ymin = p.y;
            if (p.y > ymax) ymax = p.y;
        }

        // clip
        int xMin = (int) Math.max(0, Math.min(Math.floor(xmin), w - 1));
        int xMax = (int) Math.max(0, Math.min(Math.ceil(xmax), w - 1));
        int yMin = (int) Math.max(0, Math.min(Math.floor(ymin), h - 1));
        int yMax = (int) Math.max(0, Math.min(Math.ceil(ymax), h - 1));

        if (xMax <= xMin || yMax <= yMin) {
            return 0.0f;
        }

        // 构造一个Mask
        Mat mask = Mat.zeros(yMax - yMin + 1, xMax - xMin + 1, CvType.CV_8UC1);

        // 将 box 的坐标平移
        Point[] shifted = new Point[4];
        for (int i = 0; i < 4; i++) {
            shifted[i] = new Point(box[i].x - xMin, box[i].y - yMin);
        }

        // fillPoly => 1
        MatOfPoint mop = new MatOfPoint();
        mop.fromArray(shifted);
        List<MatOfPoint> listOfContours = Collections.singletonList(mop);
        Imgproc.fillPoly(mask, listOfContours, new Scalar(1));

        // 计算 ROI 区域内的bitmap平均值
        double sumVal = 0.0;
        int count = 0;
        // 扫描 mask==1 的像素
        for (int row = 0; row < mask.rows(); row++) {
            for (int col = 0; col < mask.cols(); col++) {
                double mVal = mask.get(row, col)[0];
                if (mVal == 1) {
                    // bitmap在全局是 yMin+row, xMin+col
                    float bVal = bitmap[yMin + row][xMin + col];
                    sumVal += bVal;
                    count++;
                }
            }
        }
        if (count == 0) return 0.0f;
        return (float) (sumVal / count);
    }

    /**
     * 计算文本框平均得分(SLOW)，基于多边形本体
     */
    private float boxScoreSlow(float[][] bitmap, Point[] pts) {
        int h = bitmap.length;
        int w = bitmap[0].length;
        // contour shape (N, 2)
        // 找xmin, xmax, ymin, ymax
        double xmin = Double.MAX_VALUE, xmax = -Double.MAX_VALUE;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE;
        for (Point p : pts) {
            if (p.x < xmin) xmin = p.x;
            if (p.x > xmax) xmax = p.x;
            if (p.y < ymin) ymin = p.y;
            if (p.y > ymax) ymax = p.y;
        }

        int xMin = (int) Math.max(0, Math.min(Math.floor(xmin), w - 1));
        int xMax = (int) Math.max(0, Math.min(Math.ceil(xmax), w - 1));
        int yMin = (int) Math.max(0, Math.min(Math.floor(ymin), h - 1));
        int yMax = (int) Math.max(0, Math.min(Math.ceil(ymax), h - 1));

        if (xMax <= xMin || yMax <= yMin) {
            return 0.0f;
        }

        // 做一个mask
        Mat mask = Mat.zeros(yMax - yMin + 1, xMax - xMin + 1, CvType.CV_8UC1);

        // shift contour
        Point[] shifted = new Point[pts.length];
        for (int i = 0; i < pts.length; i++) {
            shifted[i] = new Point(pts[i].x - xMin, pts[i].y - yMin);
        }
        MatOfPoint mop = new MatOfPoint(shifted);
        List<MatOfPoint> conts = Collections.singletonList(mop);
        Imgproc.fillPoly(mask, conts, new Scalar(1));

        double sumVal = 0.0;
        int count = 0;
        for (int row = 0; row < mask.rows(); row++) {
            for (int col = 0; col < mask.cols(); col++) {
                if (mask.get(row, col)[0] == 1) {
                    float val = bitmap[yMin + row][xMin + col];
                    sumVal += val;
                    count++;
                }
            }
        }
        if (count == 0) return 0.0f;
        return (float) (sumVal / count);
    }

    /**
     * 使用Clipper进行 unclip 扩展，仿照 Python 中 pyclipper + shapely
     * 
     * @param box 原始四边形 (4个点)
     * @return 扩展后的新多边形点集
     */
    private Point[] unclip(Point[] box) {
        // 1. area/perimeter 计算
        double area = polygonArea(box);
        double perimeter = polygonPerimeter(box);
        double distance = area * this.unclipRatio / perimeter;

        // 转换为Clipper库需要的路径格式
        Path clipperPath = new Path();
        for (Point p : box) {
            clipperPath.add(new de.lighti.clipper.Point.LongPoint((long) (p.x), (long) (p.y)));
        }
        // 使用Clipper库进行扩展
        ClipperOffset offset = new ClipperOffset();
        offset.addPath(clipperPath, Clipper.JoinType.ROUND, Clipper.EndType.CLOSED_POLYGON);
        Paths expandedPaths = new Paths();
        // 计算扩展距离
        offset.execute(expandedPaths, distance);

        if (expandedPaths.isEmpty()) {
            return new Point[0]; // 返回空数组
        }
        // 取第一个扩展后的路径
        List<de.lighti.clipper.Point.LongPoint> expandedPath = expandedPaths.get(0);
        // 转换回Point数组
        Point[] result = expandedPath.stream()
                .map(p -> new Point(p.getX(), p.getY()))
                .toArray(Point[]::new);
        return result;
    }


    /**
     * 计算多边形的面积
     */
    private double polygonArea(Point[] box) {
        double area = 0.0;
        for (int i = 0; i < box.length; i++) {
            int j = (i + 1) % box.length;
            area += box[i].x * box[j].y - box[j].x * box[i].y;
        }
        return Math.abs(area / 2.0);
    }

    /**
     * 计算多边形的周长
     */
    private double polygonPerimeter(Point[] box) {
        double peri = 0.0;
        for (int i = 0; i < box.length; i++) {
            int j = (i + 1) % box.length;
            peri += dist(box[i], box[j]);
        }
        return peri;
    }

    /**
     * 用于存储 (box, minSideLen)
     */
    private static class BoxAndSize {
        Point[] box;
        float minSideLen;
        public BoxAndSize(Point[] box, float minSideLen) {
            this.box = box;
            this.minSideLen = minSideLen;
        }
    }

    /**
     * 返回结果的简单封装
     * boxes: 每个框四个点的 [ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]
     * scores: 每个框的得分
     */
    public static class ResultBundle {
        private final List<Point[]> boxes;
        private final List<Float> scores;

        public ResultBundle(List<Point[]> boxes, List<Float> scores) {
            this.boxes = boxes;
            this.scores = scores;
        }

        public List<Point[]> getBoxes() {
            return boxes;
        }

        public List<Float> getScores() {
            return scores;
        }
    }

}
