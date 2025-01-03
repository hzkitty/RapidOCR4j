package io.github.hzkitty;

import io.github.hzkitty.cal_rec_boxes.CalRecBoxes;
import io.github.hzkitty.ch_ppocr_cls.TextClassifier;
import io.github.hzkitty.ch_ppocr_det.TextDetector;
import io.github.hzkitty.ch_ppocr_rec.TextRecognizer;
import io.github.hzkitty.entity.*;
import io.github.hzkitty.utils.LoadImage;
import io.github.hzkitty.utils.OpencvLoader;
import io.github.hzkitty.utils.ProcessImg;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class RapidOCR {
    private final boolean printVerbose;
    private final float textScore;          // 过滤阈值
    private final int minHeight;            // 最小高度
    private final float widthHeightRatio;   // 宽高比
    private final int maxSideLen;           // 最大边长
    private final int minSideLen;           // 最小边长

    private final boolean useDet;           // 是否使用检测
    private final boolean useCls;           // 是否使用分类
    private final boolean useRec;           // 是否使用识别

    // OCR 三大模块和一个辅助工具
    private final TextDetector textDet;
    private final TextClassifier textCls;
    private final TextRecognizer textRec;
    private final CalRecBoxes calRecBoxes;
    private final LoadImage loadImage;


    public static RapidOCR create() {
        return new RapidOCR();
    }

    public static RapidOCR create(OcrConfig config) {
        return new RapidOCR(config);
    }

    public RapidOCR() {
        this(new OcrConfig());
    }

    public RapidOCR(OcrConfig config) {
        if (config.getGlobal().getOpencvLibPath() != null) {
            OpencvLoader.loadOpencvLib(config.getGlobal().getOpencvLibPath());
        } else {
            OpencvLoader.loadOpencvLib();
        }
        OcrConfig.GlobalConfig globalConfig = config.getGlobal();
        this.printVerbose = globalConfig.isPrintVerbose();
        this.textScore = globalConfig.getTextScore();
        this.minHeight = globalConfig.getMinHeight();
        this.widthHeightRatio = globalConfig.getWidthHeightRatio();

        // 初始化 检测/分类/识别模块
        this.useDet = globalConfig.isUseDet();
        this.textDet = new TextDetector(config.getDet());

        this.useCls = globalConfig.isUseCls();
        this.textCls = new TextClassifier(config.getCls());

        this.useRec = globalConfig.isUseRec();
        this.textRec = new TextRecognizer(config.getRec());

        this.loadImage = new LoadImage();
        this.maxSideLen = globalConfig.getMaxSideLen();
        this.minSideLen = globalConfig.getMinSideLen();

        // 初始化文本框后处理模块
        this.calRecBoxes = new CalRecBoxes();
    }

    public OcrResult run(String imagePath) throws Exception {
        return this.runImpl(imagePath, new ParamConfig());
    }

    public OcrResult run(Path imagePath) throws Exception {
        return this.runImpl(imagePath, new ParamConfig());
    }

    public OcrResult run(byte[] imageData) throws Exception {
        return this.runImpl(imageData, new ParamConfig());
    }

    public OcrResult run(BufferedImage image) throws Exception {
        return this.runImpl(image, new ParamConfig());
    }

    public OcrResult run(Mat mat) throws Exception {
        return this.runImpl(mat, new ParamConfig());
    }

    public OcrResult run(String imagePath, ParamConfig paramConfig) throws Exception {
        return this.runImpl(imagePath, paramConfig);
    }

    public OcrResult run(Path imagePath, ParamConfig paramConfig) throws Exception {
        return this.runImpl(imagePath, paramConfig);
    }

    public OcrResult run(byte[] imageData, ParamConfig paramConfig) throws Exception {
        return this.runImpl(imageData, paramConfig);
    }

    public OcrResult run(BufferedImage image, ParamConfig paramConfig) throws Exception {
        return this.runImpl(image, paramConfig);
    }

    public OcrResult run(Mat mat, ParamConfig paramConfig) throws Exception {
        return this.runImpl(mat, paramConfig);
    }

    /**
     * 执行检测 -> 分类 -> 识别 的完整流程
     *
     * @param imgContent    输入图片（支持多种格式：路径、字节数组、OpenCV Mat 等）
     * @param paramConfig   输入参数
     * @return 返回一个包含识别结果和耗时信息的自定义结构
     */
    private OcrResult runImpl(Object imgContent, ParamConfig paramConfig) throws Exception {
        long startTime = System.nanoTime(); // 记录开始时间 (纳秒)
        // 如果外部没有传值，则使用类内部的默认值
        boolean realUseDet = (paramConfig.getUseDet() == null) ? this.useDet : paramConfig.getUseDet();
        boolean realUseCls = (paramConfig.getUseCls() == null) ? this.useCls : paramConfig.getUseCls();
        boolean realUseRec = (paramConfig.getUseRec() == null) ? this.useRec : paramConfig.getUseRec();

        float realTextScore = (paramConfig.getTextScore() == null) ? this.textScore : paramConfig.getTextScore();
        boolean returnWordBox = paramConfig.getReturnWordBox() != null && paramConfig.getReturnWordBox();

        if (paramConfig.getBoxThresh() != null) {
            textDet.postprocessOp.boxThresh = paramConfig.getBoxThresh();
        }
        if (paramConfig.getUnclipRatio() != null) {
            textDet.postprocessOp.unclipRatio = paramConfig.getUnclipRatio();
        }

        // 加载图片
        Mat img = loadImage.call(imgContent);
        // 记录原始图像尺寸
        int rawH = img.rows();
        int rawW = img.cols();

        // 预处理（缩放到合理大小）
        Triple<Mat, Float, Float> processedImg = this.preprocess(img);
        img = processedImg.getLeft();
        Float ratioH = processedImg.getMiddle();
        Float ratioW = processedImg.getRight();
        Map<String, Object> opRecord = new HashMap<>();
        Map<String, Float> preprocess = new HashMap<>();
        preprocess.put("ratio_h", ratioH);
        preprocess.put("ratio_w", ratioW);
        opRecord.put("preprocess", preprocess);
        // 初始化输出结果
        List<Point[]> dtBoxes = null; // 检测到的文字区域
        List<Pair<String, Float>> clsRes = null; // 分类结果(方向, 置信度)
        List<TupleResult> recRes = null; // recRes: 每一张图对应 (text, confidence) 的列表
        double detElapsed = 0.0;
        double clsElapsed = 0.0;
        double recElapsed = 0.0;

        // ========== 1、检测阶段 ==========
        List<Mat> imgList = new ArrayList<>();
        if (realUseDet) {
            // 第二步：根据宽高比或最小高度条件，可能进行 letterbox 填充
            img = this.maybeAddLetterbox(img, opRecord);
            // 执行文本检测
            Pair<List<Point[]>, Double> detResult = textDet.call(img);
            dtBoxes = detResult.getLeft();       // 检测得到的文本框
            detElapsed = detResult.getRight();  // 检测所耗时间（秒）

            if (dtBoxes == null || dtBoxes.isEmpty()) {
                // 如果没检测到任何文本框，直接返回空结果
                double elapseSec = (System.nanoTime() - startTime) / 1e9; // 转为秒
                return new OcrResult("", Collections.emptyList(), elapseSec, detElapsed, clsElapsed, recElapsed);
            }

            // 对检测结果进行排序
            dtBoxes = this.sortedBoxes(dtBoxes);

            // 裁剪出检测到的文本框区域
            imgList = this.getCropImgList(img, dtBoxes);
        } else {
            imgList.add(img);
        }

        // ========== 2、分类阶段 ==========
        if (realUseCls) {
            Triple<List<Mat>, List<Pair<String, Float>>, Double> clsResultTriple = textCls.call(imgList);
            imgList = clsResultTriple.getLeft();  // 分类器可能帮我们转正图像
            clsRes = clsResultTriple.getMiddle();          // 分类的标签+置信度
            clsElapsed = clsResultTriple.getRight();       // 分类耗时(秒)
        }

        // ========== 3、识别阶段 ==========
        if (realUseRec) {
            // 是否返回单词级别的框
            Pair<List<TupleResult>, Double> resultBundle = textRec.call(imgList, returnWordBox);
            recRes = resultBundle.getLeft();
            recElapsed = resultBundle.getRight();
        }

        // ========== 后处理：计算 word-level boxes（可选） ==========
        if (dtBoxes != null && recRes != null && returnWordBox) {
            // 调用 calRecBoxes
            recRes = calRecBoxes.call(imgList, dtBoxes, recRes);

            // 接下来需要把坐标映射回原图 (根据预处理记录 + Padding 记录)
            for (TupleResult recResi : recRes) {
                if (recResi.getWordBoxResult().getSortedWordBoxList() != null) {
                    List<Point[]> originPoints = this.getOriginPoints(recResi.getWordBoxResult().getSortedWordBoxList(), opRecord, rawH, rawW);
                    recResi.getWordBoxResult().setSortedWordBoxList(originPoints);
                }
            }
        }

        if (dtBoxes != null) {
            dtBoxes = this.getOriginPoints(dtBoxes, opRecord, rawH, rawW);
        }

        double elapseSec = (System.nanoTime() - startTime) / 1e9; // 转为秒
        return getFinalRes(dtBoxes, clsRes, recRes, elapseSec, detElapsed, clsElapsed, recElapsed);
    }

    public OcrResult getFinalRes(List<Point[]> dtBoxes, List<Pair<String, Float>> clsRes, List<TupleResult> recRes,
                                 double elapseSec, double detElapsed, double clsElapsed, double recElapsed) {
        if (dtBoxes == null && recRes == null && clsRes != null) {
            // 将 clsRes 转成 RecResult 列表
            List<RecResult> recResultList = new ArrayList<>();
            for (Pair<String, Float> pair : clsRes) {
                recResultList.add(new RecResult(null, pair.getLeft(), pair.getRight(), null));
            }
            return new OcrResult("", recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
        }

        if (dtBoxes == null && recRes == null) {
            return new OcrResult("", Collections.emptyList(), elapseSec, detElapsed, clsElapsed, recElapsed);
        }

        if (dtBoxes == null && recRes != null) {
            // 将 recRes 封装到 RecResult
            List<RecResult> recResultList = new ArrayList<>();
            for (TupleResult tr : recRes) {
                recResultList.add(new RecResult(null, tr.getText(), tr.getConfidence(), tr.getWordBoxResult()));
            }
            String strRes = recResultList.stream().map(RecResult::getText).collect(Collectors.joining("\n"));
            return new OcrResult(strRes, recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
        }

        if (dtBoxes != null && recRes == null) {
            // 将 dtBoxes 简单封装到 RecResult
            List<RecResult> recResultList = new ArrayList<>();
            for (Point[] box : dtBoxes) {
                recResultList.add(new RecResult(box, "", 0.0f, null));
            }
            return new OcrResult("", recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
        }
        Pair<List<Point[]>, List<TupleResult>> filtered = filterResult(dtBoxes, recRes);
        List<Point[]> filteredBoxes = filtered.getLeft();
        List<TupleResult> filteredRes = filtered.getRight();
        if (filteredBoxes == null || filteredRes == null || filteredBoxes.isEmpty()) {
            return new OcrResult("", Collections.emptyList(), elapseSec, detElapsed, clsElapsed, recElapsed);
        }
        // 封装 RecResult
        List<RecResult> recResultList = new ArrayList<>();
        for (int i = 0; i < filteredBoxes.size(); i++) {
            TupleResult tr = filteredRes.get(i);
            RecResult recResult = new RecResult(filteredBoxes.get(i), tr.getText(),
                    tr.getConfidence(), tr.getWordBoxResult());
            recResultList.add(recResult);
        }
        String strRes = recResultList.stream().map(RecResult::getText).collect(Collectors.joining("\n"));
        return new OcrResult(strRes, recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
    }

    public Pair<List<Point[]>, List<TupleResult>> filterResult(List<Point[]> dtBoxes, List<TupleResult> recRes) {
        if (dtBoxes == null || recRes == null) {
            return Pair.of(null, null);
        }
        List<Point[]> filterBoxes = new ArrayList<>();
        List<TupleResult> filterRecRes = new ArrayList<>();

        for (int i = 0; i < dtBoxes.size() && i < recRes.size(); i++) {
            Point[] box = dtBoxes.get(i);
            TupleResult t = recRes.get(i);

            // if score >= self.text_score
            if (t.getConfidence() >= textScore) {
                filterBoxes.add(box);
                filterRecRes.add(t);
            }
        }
        return Pair.of(filterBoxes, filterRecRes);
    }

    /**
     * 预处理：先判断图像最大边是否超过 maxSideLen，如果是则等比例缩小；
     * 再判断图像最小边是否小于 minSideLen，如果是则等比例放大。
     */
    private Triple<Mat, Float, Float> preprocess(Mat img) {
        int h = img.rows();
        int w = img.cols();
        int maxValue = Math.max(h, w);

        float ratioH = 1.0f;
        float ratioW = 1.0f;

        // 如果最大边超出限制则缩小
        if (maxValue > this.maxSideLen) {
            try {
                Triple<Mat, Float, Float> reduced = ProcessImg.reduceMaxSide(img, this.maxSideLen);
                img = reduced.getLeft();
                ratioH = reduced.getMiddle();
                ratioW = reduced.getRight();
            } catch (ProcessImg.ResizeImgError e) {
                e.printStackTrace();
            }
        }

        // 更新 h/w
        h = img.rows();
        w = img.cols();
        int minValue = Math.min(h, w);

        // 如果最小边小于限制则放大
        if (minValue < this.minSideLen) {
            try {
                Triple<Mat, Float, Float> increased = ProcessImg.increaseMinSide(img, this.minSideLen);
                Mat after = increased.getLeft();
                float scaleH = increased.getMiddle();
                float scaleW = increased.getRight();

                // 注意：要将前一次 ratio 乘上这次的 ratio 才是最终总缩放比
                ratioH *= scaleH;
                ratioW *= scaleW;
                img = after;
            } catch (ProcessImg.ResizeImgError e) {
                e.printStackTrace();
            }
        }

        return Triple.of(img, ratioH, ratioW);
    }

    /**
     * 判断是否需要添加 letterbox（当 h <= minHeight 或者 宽高比超过设定阈值时）
     */
    private Mat maybeAddLetterbox(Mat img, Map<String, Object> opRecord) {
        int h = img.rows();
        int w = img.cols();

        boolean useLimitRatio = true;
        if (this.widthHeightRatio == -1.0f) {
            useLimitRatio = false;
        }

        Map<String, Integer> padding = new HashMap<>();
        if (h <= this.minHeight || (useLimitRatio && (float) w / (float) h > this.widthHeightRatio)) {
            int paddingH = getPaddingH(h, w);
            Mat blockImg = ProcessImg.addRoundLetterbox(img, paddingH, paddingH, 0, 0);

            // 记录一下填充量，用于后面逆变换
            padding.put("top", paddingH);
            padding.put("left", 0);
            opRecord.put("padding", padding);
            return blockImg;
        }

        // 否则不做填充
        padding.put("top", 0);
        padding.put("left", 0);
        opRecord.put("padding", padding);
        return img;
    }

    /**
     * 获取需要在高度方向上填充的值
     */
    private int getPaddingH(int h, int w) {
        float targetH = Math.max((w / widthHeightRatio), (float) minHeight);
        int newH = (int) targetH * 2;
        return Math.abs(newH - h) / 2;
    }

    /**
     * 对检测结果进行排序：从上到下，再从左到右
     * 每个框为 Point[4]
     */
    private List<Point[]> sortedBoxes(List<Point[]> dtBoxes) {
        // 先按照第一个点的 y, x 排序
        dtBoxes.sort((o1, o2) -> {
            double y1 = o1[0].y;
            double y2 = o2[0].y;
            double x1 = o1[0].x;
            double x2 = o2[0].x;
            if (Double.compare(y1, y2) == 0) {
                return Double.compare(x1, x2);
            } else {
                return Double.compare(y1, y2);
            }
        });

        // 再做相邻交换，如果 y 相差较小，但 x 前后顺序相反，则交换
        for (int i = 0; i < dtBoxes.size() - 1; i++) {
            for (int j = i; j >= 0; j--) {
                double yDiff = Math.abs(dtBoxes.get(j + 1)[0].y - dtBoxes.get(j)[0].y);
                if (yDiff < 10 && dtBoxes.get(j + 1)[0].x < dtBoxes.get(j)[0].x) {
                    // 交换
                    Collections.swap(dtBoxes, j, j + 1);
                } else {
                    break;
                }
            }
        }
        return dtBoxes;
    }

    /**
     * 根据检测框列表裁剪图像
     *
     * @param img     原始图像
     * @param dtBoxes 检测框列表，每个检测框由四个顶点组成
     * @return 裁剪后的图像列表
     */
    private List<Mat> getCropImgList(Mat img, List<Point[]> dtBoxes) {
        // 使用Stream流处理每个检测框并裁剪图像
        return dtBoxes.stream()
                .map(box -> getRotateCropImage(img, box))
                .collect(Collectors.toList());
    }

    /**
     * 根据旋转框的四个顶点裁剪图像
     *
     * @param img    原始图像
     * @param points 检测框的四个顶点
     * @return 裁剪后的图像
     */
    private Mat getRotateCropImage(Mat img, Point[] points) {
        // 计算裁剪区域的宽度
        double widthTop = distance(points[0], points[1]);
        double widthBottom = distance(points[2], points[3]);
        int imgCropWidth = (int) Math.max(widthTop, widthBottom);

        // 计算裁剪区域的高度
        double heightLeft = distance(points[0], points[3]);
        double heightRight = distance(points[1], points[2]);
        int imgCropHeight = (int) Math.max(heightLeft, heightRight);

        // 定义标准的裁剪区域四个顶点
        MatOfPoint2f ptsStd = new MatOfPoint2f(
                new Point(0, 0),
                new Point(imgCropWidth, 0),
                new Point(imgCropWidth, imgCropHeight),
                new Point(0, imgCropHeight)
        );

        // 将检测框顶点转换为MatOfPoint2f
        MatOfPoint2f ptsSrc = new MatOfPoint2f(points);

        // 计算透视变换矩阵
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(ptsSrc, ptsStd);

        // 执行透视变换裁剪图像
        Mat dstImg = new Mat();
        Imgproc.warpPerspective(img, dstImg, perspectiveTransform,
                new Size(imgCropWidth, imgCropHeight),
                Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE);

        // 判断裁剪后的图像是否需要旋转
        if (((double) dstImg.rows() / dstImg.cols()) >= 1.5) {
            Core.rotate(dstImg, dstImg, Core.ROTATE_90_CLOCKWISE);
        }

        return dstImg;
    }

    /**
     * 计算两点之间的欧几里得距离
     *
     * @param p1 第一个点
     * @param p2 第二个点
     * @return 两点之间的距离
     */
    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    /**
     * 把坐标映射回原图，使用 PreProcessRecord 和 PaddingRecord 作为传参对象存储预处理信息。
     *
     * @param dtBoxes  : 检测框列表，每个元素是一个长度为4的 Point[]
     * @param opRecord : 记录了一系列图像预处理操作及其参数的 Map<String, Object>
     *                 其中 key 为操作名称，例如 "padding"、"preprocess"，
     *                 value 可能是 Map<String, Integer> 或 Map<String, Float>。
     * @param rawH     : 原图的高度
     * @param rawW     : 原图的宽度
     * @return 修正后的坐标列表 (同一个对象内坐标已被就地修改)
     */
    private List<Point[]> getOriginPoints(List<Point[]> dtBoxes, Map<String, Object> opRecord, int rawH, int rawW) {
        // 1. 逆序遍历 opRecord 的键
        List<String> keys = new ArrayList<>(opRecord.keySet());
        Collections.reverse(keys);

        // 2. 针对每个预处理操作，做相应的逆变换
        for (String op : keys) {
            Object recordObj = opRecord.get(op);
            if (recordObj == null) {
                continue; // 如果没有该 key 对应的对象，跳过
            }

            // 2.1 如果是 padding
            if (op.contains("padding")) {
                Map<String, Integer> padding = (Map<String, Integer>) recordObj;
                int top = padding.get("top");
                int left = padding.get("left");

                // 对每个检测框的每个点进行偏移还原
                for (Point[] box : dtBoxes) {
                    for (Point p : box) {
                        p.x -= left;
                        p.y -= top;
                    }
                }
            }
            // 2.2 如果是 preprocess(一般指缩放)
            else if (op.contains("preprocess")) {
                Map<String, Float> preprocess = (Map<String, Float>) recordObj;
                float ratioH = preprocess.get("ratio_h");
                float ratioW = preprocess.get("ratio_w");

                // 对每个检测框的每个点进行缩放还原
                for (Point[] box : dtBoxes) {
                    for (Point p : box) {
                        p.x *= ratioW;
                        p.y *= ratioH;
                    }
                }
            }
        }

        // 3. 将所有坐标裁剪到 [0, rawW] 和 [0, rawH]
        for (Point[] box : dtBoxes) {
            for (Point p : box) {
                if (p.x < 0) {
                    p.x = 0;
                } else if (p.x > rawW) {
                    p.x = rawW;
                }
                if (p.y < 0) {
                    p.y = 0;
                } else if (p.y > rawH) {
                    p.y = rawH;
                }
            }
        }

        // 4. 返回修正后的坐标（dtBoxes 已经就地修改）
        return dtBoxes;
    }

}
