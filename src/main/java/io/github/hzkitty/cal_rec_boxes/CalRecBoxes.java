package io.github.hzkitty.cal_rec_boxes;

import io.github.hzkitty.entity.TupleResult;
import io.github.hzkitty.entity.WordBoxInfo;
import io.github.hzkitty.entity.WordBoxResult;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * CalRecBoxes 主要用于根据OCR识别结果，计算单词/汉字在原图上的坐标框
 * 适用于类似于 OCR 检测识别后，根据识别文本的列索引，去拆分出每一个单字(汉字)
 * 或者单词(英文)的坐标框。并且对倾斜、旋转等进行校正。
 */
public class CalRecBoxes {

    /**
     * 主入口：根据输入的图像、检测框以及OCR的识别结果，计算每行文字对应的子字符/单词坐标
     *
     * @param imgs    原图列表
     * @param dtBoxes 对应的检测框列表，每个元素是一个长度为4的Point[]，顺序：左上、右上、右下、左下
     * @param recRes  OCR识别结果，每个元素封装在 TupleResult 中
     * @return 返回新的TupleResult列表，每个元素里可包含最终计算后的字符/单词坐标等信息
     */
    public List<TupleResult> call(List<Mat> imgs, List<Point[]> dtBoxes, List<TupleResult> recRes) {
        List<TupleResult> results = new ArrayList<>();

        for (int i = 0; i < imgs.size(); i++) {
            Mat img = imgs.get(i);
            Point[] box = dtBoxes.get(i);
            TupleResult tupleRes = recRes.get(i);

            // 1. 获取方向 (水平/垂直)
            String direction = getBoxDirection(box);

            // 2. 获取OCR识别文本、置信度、以及字/单词信息
            String recTxt = tupleRes.getText();
            float recConf = tupleRes.getConfidence();
            WordBoxInfo recWordInfo = tupleRes.getWordBoxInfo();

            // 3. 构造整段文字的最大矩形边界
            int h = img.rows();
            int w = img.cols();
            // imgBox: 左上、右上、右下、左下
            Point[] imgBox = new Point[4];
            imgBox[0] = new Point(0, 0);
            imgBox[1] = new Point(w, 0);
            imgBox[2] = new Point(w, h);
            imgBox[3] = new Point(0, h);

            // 4. 计算单词（或单字）坐标框
            WordBoxResult boxResult = calOcrWordBox(recTxt, imgBox, recWordInfo);

            // 5. 调整坐标框(去除重叠)
            List<Point[]> adjustedBoxList = adjustBoxOverlap(boxResult.getSortedWordBoxList());

            // 6. 根据dtBox(实际检测框) + 方向，进行坐标的逆映射(校正)
            List<Point[]> finalBoxList = reverseRotateCropImage(box, adjustedBoxList, direction);

            // 7. 将计算后的坐标信息保存到新的 TupleResult 中
            TupleResult newResult = new TupleResult(recTxt, recConf, null);
            WordBoxResult wordBoxResult = new WordBoxResult(boxResult.getWordBoxContentList(),
                    finalBoxList,
                    boxResult.getConfList());
            newResult.setWordBoxResult(wordBoxResult);
            results.add(newResult);
        }

        return results;
    }

    /**
     * 根据一个检测框判断是水平还是竖直
     * 若高度/宽度 >= 1.5，则判定为竖直(“h”)，否则为水平(“w”)
     */
    private String getBoxDirection(Point[] box) {
        double width1 = distance(box[0], box[1]);
        double width2 = distance(box[2], box[3]);
        double imgCropWidth = Math.max(width1, width2);

        double height1 = distance(box[0], box[3]);
        double height2 = distance(box[1], box[2]);
        double imgCropHeight = Math.max(height1, height2);

        if (imgCropHeight / imgCropWidth >= 1.5) {
            return "h";
        }
        return "w";
    }

    /**
     * 计算OCR中每个字(汉字)或每个单词(英文)的坐标框
     *
     * @param recTxt      识别文本
     * @param box         对应整块文本的矩形(4点)
     * @param recWordInfo 其中存储列数、字/词列表、每个字/词对应的列索引等
     */
    private WordBoxResult calOcrWordBox(String recTxt, Point[] box, WordBoxInfo recWordInfo) {
        //  col_num, word_list, word_col_list, state_list, conf_list = rec_word_info
        double colNum = recWordInfo.getTextIndexLen();  // 列数
        List<List<String>> wordList = recWordInfo.getWordList();
        List<List<Integer>> wordColList = recWordInfo.getWordColList();
        List<String> stateList = recWordInfo.getStateList();
        List<Float> confList = recWordInfo.getConfList();

        // box: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        double bboxXStart = box[0].x;
        double bboxXEnd = box[1].x;
        double bboxYStart = box[0].y;
        double bboxYEnd = box[2].y;

        double cellWidth = (bboxXEnd - bboxXStart) / colNum;

        List<Point[]> wordBoxList = new ArrayList<>();
        List<String> wordBoxContentList = new ArrayList<>();

        // 用来统计平均字符宽度的容器
        List<Double> cnWidthList = new ArrayList<>();
        List<Double> enWidthList = new ArrayList<>();
        List<Integer> cnColList = new ArrayList<>();
        List<Integer> enColList = new ArrayList<>();

        // 1. 遍历每个单词(或汉字)，区分中英文，统计宽度
        for (int i = 0; i < wordList.size(); i++) {
            List<String> word = wordList.get(i);
            List<Integer> wordCol = wordColList.get(i);
            String state = stateList.get(i);

            if ("cn".equals(state)) {
                calCharWidth(cnWidthList, wordCol, cellWidth);
                cnColList.addAll(wordCol);
                wordBoxContentList.addAll(word);
            } else { // 英文
                calCharWidth(enWidthList, wordCol, cellWidth);
                enColList.addAll(wordCol);
                wordBoxContentList.addAll(word);
            }
        }

        // 2. 根据中英文各自的平均字符宽度，计算最终坐标
        calBox(cnColList, cnWidthList, wordBoxList, bboxXStart, bboxXEnd, bboxYStart, bboxYEnd, recTxt.length(), cellWidth);
        calBox(enColList, enWidthList, wordBoxList, bboxXStart, bboxXEnd, bboxYStart, bboxYEnd, recTxt.length(), cellWidth);

        // 3. 按 x 坐标排序
        wordBoxList.sort(Comparator.comparingDouble(o -> o[0].x));

        // 返回结果
        return new WordBoxResult(wordBoxContentList, wordBoxList, confList);
    }

    /**
     * 计算单个字/词的统计宽度
     */
    private void calCharWidth(List<Double> widthList, List<Integer> wordCol, double cellWidth) {
        // 若只有一个列索引，则无法计算(它只能是单独一个字符或一个词)
        if (wordCol.size() <= 1) {
            return;
        }
        // 计算字符总长度
        double charTotalLength = (wordCol.get(wordCol.size() - 1) - wordCol.get(0)) * cellWidth;
        double charWidth = charTotalLength / (wordCol.size() - 1);
        widthList.add(charWidth);
    }

    /**
     * 根据colList和统计得到的平均宽度，计算坐标框并存入wordBoxList
     */
    private void calBox(
            List<Integer> colList,
            List<Double> widthList,
            List<Point[]> wordBoxList,
            double bboxXStart,
            double bboxXEnd,
            double bboxYStart,
            double bboxYEnd,
            int recTxtLen,
            double cellWidth
    ) {
        if (colList.isEmpty()) {
            return;
        }
        // 如果有统计的字符宽度，取均值，否则就用 (总宽 / 字符数) 兜底
        double avgCharWidth = 0.0;
        if (!widthList.isEmpty()) {
            avgCharWidth = widthList.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        } else {
            avgCharWidth = (bboxXEnd - bboxXStart) / Math.max(1, recTxtLen);
        }

        for (Integer centerIdx : colList) {
            double centerX = (centerIdx + 0.5) * cellWidth;
            double halfW = avgCharWidth / 2.0;
            double cellXStart = Math.max(Math.floor(centerX - halfW), 0) + bboxXStart;
            double cellXEnd = Math.min(Math.floor(centerX + halfW), (bboxXEnd - bboxXStart)) + bboxXStart;

            Point[] cell = new Point[]{
                    new Point(cellXStart, bboxYStart),
                    new Point(cellXEnd, bboxYStart),
                    new Point(cellXEnd, bboxYEnd),
                    new Point(cellXStart, bboxYEnd)
            };
            wordBoxList.add(cell);
        }
    }

    /**
     * 调整bbox有重叠的地方
     */
    private List<Point[]> adjustBoxOverlap(List<Point[]> wordBoxList) {
        for (int i = 0; i < wordBoxList.size() - 1; i++) {
            Point[] cur = wordBoxList.get(i);
            Point[] nxt = wordBoxList.get(i + 1);

            // 判断 cur[1].x > nxt[0].x 说明出现左右交叠
            if (cur[1].x > nxt[0].x) {
                double distance = Math.abs(cur[1].x - nxt[0].x);
                double halfDist = distance / 2.0;

                // cur 右上、右下往左缩
                cur[1].x -= halfDist;
                cur[2].x -= halfDist;

                // nxt 左上、左下往右缩
                nxt[0].x += (distance - halfDist);
                nxt[3].x += (distance - halfDist);
            }
        }
        return wordBoxList;
    }

    /**
     * 逆操作：将文字在crop后坐标映射回原图的坐标
     *
     * @param bboxPoints     crop区域在原图上的4点
     * @param wordPointsList 在crop图中的若干字(词)坐标
     * @param direction      "h" 或 "w"
     */
    private List<Point[]> reverseRotateCropImage(Point[] bboxPoints, List<Point[]> wordPointsList, String direction) {
        // 1. 计算 bboxPoints 最小 x、y，以便后续平移到 (0,0)
        double left = Math.min(
                Math.min(bboxPoints[0].x, bboxPoints[3].x),
                Math.min(bboxPoints[1].x, bboxPoints[2].x)
        );
        double top = Math.min(
                Math.min(bboxPoints[0].y, bboxPoints[1].y),
                Math.min(bboxPoints[2].y, bboxPoints[3].y)
        );

        // 2. 将 bboxPoints 平移，保证左上角为 (0,0)
        Point[] shiftBox = new Point[4];
        for (int i = 0; i < 4; i++) {
            shiftBox[i] = new Point(bboxPoints[i].x - left, bboxPoints[i].y - top);
        }

        double imgCropWidth = distance(shiftBox[0], shiftBox[1]);
        double imgCropHeight = distance(shiftBox[0], shiftBox[3]);

        // 3. 构建透视变换矩阵 M (从 crop 后坐标 -> "正矩形"的坐标)
        MatOfPoint2f srcPoints = new MatOfPoint2f(
                shiftBox[0], shiftBox[1], shiftBox[2], shiftBox[3]
        );
        MatOfPoint2f dstPoints = new MatOfPoint2f(
                new Point(0, 0),
                new Point(imgCropWidth, 0),
                new Point(imgCropWidth, imgCropHeight),
                new Point(0, imgCropHeight)
        );
        Mat M = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);

        // 4. 计算逆矩阵 IM (用于从 "正矩形坐标" -> 回到原图坐标)
        Mat IM = M.inv();

        // 5. 对每个 wordPoints 执行逆变换
        List<Point[]> newWordPointsList = new ArrayList<>();
        for (Point[] wordPoints : wordPointsList) {
            // 用于存储当前字(词)的4个点(或更多)变换结果
            Point[] transformedPoints = new Point[wordPoints.length];

            for (int i = 0; i < wordPoints.length; i++) {
                Point pt = wordPoints[i];

                // 5.1 若文本是竖直方向("h")，需先做一次顺时针 -90° 旋转修正
                Point toRotate = pt;
                if ("h".equals(direction)) {
                    toRotate = sRotate(Math.toRadians(-90.0), pt.x, pt.y, 0, 0);
                    // 旋转后再在水平方向上平移 imgCropWidth
                    toRotate.x += imgCropWidth;
                }

                // 5.2 做逆透视变换： (x', y', 1) = IM * (x, y, 1)
                Mat vec = Mat.zeros(3, 1, CvType.CV_64F);
                vec.put(0, 0, toRotate.x);
                vec.put(1, 0, toRotate.y);
                vec.put(2, 0, 1.0);

                Mat dst = new Mat();
                Core.gemm(IM, vec, 1.0, new Mat(), 0.0, dst);
                double x = dst.get(0, 0)[0];
                double y = dst.get(1, 0)[0];
                double z = dst.get(2, 0)[0];
                double nx = x / z;
                double ny = y / z;

                // 5.3 将平移补偿加回去
                nx += left;
                ny += top;

                transformedPoints[i] = new Point(nx, ny);
            }

            // 5.4 对新的坐标点排序 (选用自定义 orderPoints)
            //     如果确定每个框只有 4 点、并且有固定顺序，可不再排序
            Point[] ordered = orderPoints(transformedPoints);

            newWordPointsList.add(ordered);
        }

        return newWordPointsList;
    }

    /**
     * 绕 (pointx, pointy) 顺时针旋转 angle
     */
    private Point sRotate(double angle, double valuex, double valuey, double pointx, double pointy) {
        double sRotatex = (valuex - pointx) * Math.cos(angle)
                + (valuey - pointy) * Math.sin(angle)
                + pointx;
        double sRotatey = (valuey - pointy) * Math.cos(angle)
                - (valuex - pointx) * Math.sin(angle)
                + pointy;
        return new Point(sRotatex, sRotatey);
    }

    /**
     * 对矩形的四点进行排序：左上、右上、右下、左下
     * 如果是菱形或其他情况，则做相应的特殊处理
     */
    private Point[] orderPoints(Point[] box) {
        if (box == null || box.length < 4) {
            // 如果不足4点，直接返回原数组
            return box;
        }

        // 求中心点
        double sumX = 0.0;
        double sumY = 0.0;
        for (Point p : box) {
            sumX += p.x;
            sumY += p.y;
        }
        double centerX = sumX / box.length;
        double centerY = sumY / box.length;

        // 按照与中心的极角从小到大排序
        Arrays.sort(box, (p1, p2) -> {
            double angle1 = Math.atan2(p1.y - centerY, p1.x - centerX);
            double angle2 = Math.atan2(p2.y - centerY, p2.x - centerX);
            return Double.compare(angle1, angle2);
        });

        return box;
    }

    /**
     * 计算两点距离
     */
    private double distance(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

}
