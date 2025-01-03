package org.hzkitty.ch_ppocr_cls;

import ai.onnxruntime.OrtException;
import org.hzkitty.entity.Pair;
import org.hzkitty.entity.Triple;
import org.hzkitty.entity.OcrConfig;
import org.hzkitty.entity.OrtInferConfig;
import org.hzkitty.utils.OrtInferSession;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * 对输入图像进行文字方向分类，并在需要时旋转图像。
 */
public class TextClassifier {

    // 分类输入图像的形状 [channels, height, width]
    private final int[] clsImageShape;
    // 分类批量处理数量
    private final int clsBatchNum;
    // 分类阈值
    private final float clsThresh;
    // 分类后处理
    private final ClsPostProcess postProcessOp;
    // ONNX 运行会话
    private final OrtInferSession inferSession;

    /**
     * 构造函数：从 OCRConfig 中读取分类相关的配置，并进行初始化。
     *
     * @param clsConfig OCRConfig，其中包含了分类模块 ClsConfig 的相关配置
     */
    public TextClassifier(OcrConfig.ClsConfig clsConfig) {
        // 从配置中读取相关的分类参数
        this.clsImageShape = clsConfig.getClsImageShape(); // [C, H, W]
        this.clsBatchNum = clsConfig.getClsBatchNum();
        this.clsThresh = clsConfig.getClsThresh();
        this.postProcessOp = new ClsPostProcess(clsConfig.getLabelList());

        // 初始化推理会话
        OrtInferConfig ortInferConfig = new OrtInferConfig();
        ortInferConfig.setIntraOpNumThreads(clsConfig.getIntraOpNumThreads());
        ortInferConfig.setInterOpNumThreads(clsConfig.getInterOpNumThreads());
        ortInferConfig.setUseCuda(clsConfig.isUseArena());
        ortInferConfig.setUseDml(clsConfig.isUseDml());
        ortInferConfig.setModelPath(clsConfig.getModelPath());
        ortInferConfig.setUseArena(clsConfig.isUseArena());
        this.inferSession = new OrtInferSession(ortInferConfig);
    }

    /**
     * 对输入图像列表进行分类，若识别到 180 度的标签且分数超过阈值，则进行图像旋转。
     *
     * @param imgList 待分类的图像列表（OpenCV Mat 格式）
     * @return 三元组：(1) 处理后的图像列表（可能会被旋转）；(2) [标签, 分数] 列表；(3) 运行所耗时(秒)
     */
    public Triple<List<Mat>, List<Pair<String, Float>>, Double> call(List<Mat> imgList) {
        // 记录开始时间
        long start = System.currentTimeMillis();

        // 计算每张图像的宽高比，用于排序优化
        List<Double> widthRatioList = new ArrayList<>();
        for (Mat img : imgList) {
            double h = img.rows();
            double w = img.cols();
            widthRatioList.add(w / h);
        }

        // 获取排序后的索引序列
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < imgList.size(); i++) {
            indices.add(i);
        }
        // 按照宽高比升序排序
        indices.sort(Comparator.comparingDouble(widthRatioList::get));

        // 准备存放分类结果 [label, score]
        List<Pair<String, Float>> clsRes = new ArrayList<>();
        for (int i = 0; i < imgList.size(); i++) {
            // 先填充空结果，保持位置对齐
            clsRes.add(Pair.of("", 0.0f));
        }

        int imgNum = imgList.size();

        // 分批处理
        for (int beg = 0; beg < imgNum; beg += clsBatchNum) {
            int end = Math.min(imgNum, beg + clsBatchNum);

            // 构建当前 batch 的预处理结果
            List<float[][][][]> batchDataList = new ArrayList<>();

            for (int idx = beg; idx < end; idx++) {
                int realIndex = indices.get(idx);
                // resize & normalize
                float[][][] normImg = resizeNormImg(imgList.get(realIndex));
                // 因为推理输入 shape 通常为 [N, C, H, W]，故外层再包一层
                float[][][][] oneInput = new float[1][normImg.length][normImg[0].length][normImg[0][0].length];
                oneInput[0] = normImg;
                batchDataList.add(oneInput);
            }

            // 将 batchDataList 里的所有样本合并为一个批次 [batchSize, C, H, W]
            float[][][][] batchData = concatBatchData(batchDataList);

            // 推理
            float[][] probOut;
            try {
                probOut = (float[][]) inferSession.run(batchData);
            } catch (OrtException e) {
                e.printStackTrace();
                continue;
            }
            // 后处理，得到每个样本的 [label, score]
            List<Pair<String, Float>> clsResult = postProcessOp.call(probOut);

            // 将分类结果放回原来的顺序位置
            for (int rno = 0; rno < clsResult.size(); rno++) {
                Pair<String, Float> pair = clsResult.get(rno);
                String label = pair.getLeft();
                float score = pair.getRight();

                // 放到 clsRes 对应的真实索引位置中
                int realIndex = indices.get(beg + rno);
                clsRes.set(realIndex, Pair.of(label, score));

                // 如果预测为 “180” 且置信度超过阈值，则在原图上进行 180 度旋转
                if (label.contains("180") && score > clsThresh) {
                    // OpenCV 旋转 180 度
                    Mat original = imgList.get(realIndex);
                    Core.rotate(original, original, Core.ROTATE_180);
                }
            }
        }

        // 计算总耗时（秒）
        long end = System.currentTimeMillis();
        double totalSeconds = (end - start) / 1000.0;

        // 返回处理过后的图像列表、分类结果以及耗时
        return Triple.of(imgList, clsRes, totalSeconds);
    }

    /**
     * 将单张图像进行 resize 和归一化处理，返回 float[C][H][W]。
     *
     * @param img OpenCV Mat 格式的单张图像
     * @return 归一化后的图像数据 float[C][H][W]
     */
    private float[][][] resizeNormImg(Mat img) {
        // clsImageShape: [channels, imgH, imgW]
        int imgC = clsImageShape[0];
        int imgH = clsImageShape[1];
        int imgW = clsImageShape[2];

        int h = img.rows();
        int w = img.cols();
        float ratio = (float) w / (float) h;

        // 根据长宽比，判断 resize 后的宽度
        int resizedW = (int) Math.ceil(imgH * ratio);
        if (resizedW > imgW) {
            resizedW = imgW;
        }

        // 1) 先把图像 resize 到 (resizedW, imgH)
        Mat resizedMat = new Mat();
        Imgproc.resize(img, resizedMat, new Size(resizedW, imgH));

        // 如果只有单通道
        if (imgC == 1 && resizedMat.channels() == 3) {
            // 这里可以改成灰度化，或者根据你实际的模型需要来处理
            Imgproc.cvtColor(resizedMat, resizedMat, Imgproc.COLOR_BGR2GRAY);
        }

        // 2) 转换为 float 类型，并除以 255
        resizedMat.convertTo(resizedMat, CvType.CV_32FC(resizedMat.channels()), 1.0 / 255.0);

        // 3) 减均值 0.5，再除以 0.5
        Core.subtract(resizedMat, new Mat(resizedMat.size(), resizedMat.type(),
                        resizedMat.channels() == 1 ? new org.opencv.core.Scalar(0.5)
                                : new org.opencv.core.Scalar(0.5, 0.5, 0.5)),
                resizedMat);
        Core.divide(resizedMat, new Mat(resizedMat.size(), resizedMat.type(),
                        resizedMat.channels() == 1 ? new org.opencv.core.Scalar(0.5)
                                : new org.opencv.core.Scalar(0.5, 0.5, 0.5)),
                resizedMat);

        // 4) 在 (imgH, imgW) 的画布上进行右侧 padding
        //    创建一个 (imgH, imgW) 大小，通道数为 imgC，初始值为 0.0 的 Mat
        Mat paddingMat = new Mat(imgH, imgW, CvType.CV_32FC(resizedMat.channels()));
        paddingMat.setTo(new org.opencv.core.Scalar(0.0, 0.0, 0.0));

        // 将 resizedMat 拷贝到 paddingMat 的左上角
        Mat subArea = paddingMat.submat(0, imgH, 0, resizedW);
        resizedMat.copyTo(subArea);

        // 5) 将 paddingMat 转为 float[C][H][W]，以符合推理输入
        float[][][] chwData = matToCHWFloatArray(paddingMat, imgC, imgH, imgW);

        return chwData;
    }

    /**
     * 将 [1, C, H, W] 的若干数据合并为 [batchSize, C, H, W]。
     *
     * @param batchDataList 各样本的数据列表
     * @return 合并后的四维数组
     */
    private float[][][][] concatBatchData(List<float[][][][]> batchDataList) {
        int batchSize = batchDataList.size();
        // 假设都符合 [1, C, H, W] 形状
        int c = batchDataList.get(0)[0].length;
        int h = batchDataList.get(0)[0][0].length;
        int w = batchDataList.get(0)[0][0][0].length;

        float[][][][] out = new float[batchSize][c][h][w];
        for (int i = 0; i < batchSize; i++) {
            // 因为 batchDataList.get(i) 是一个 [1, C, H, W] 的数组
            float[][][] data = batchDataList.get(i)[0];
            out[i] = data;
        }
        return out;
    }

    /**
     * 将 OpenCV Mat 转为形状为 float[C][H][W] 的三维数组 (CHW 格式)。
     *
     * @param mat 原始 Mat 数据 (float 类型)
     * @param c   通道数
     * @param h   高
     * @param w   宽
     * @return CHW 格式的三维数组 float[C][H][W]
     */
    private float[][][] matToCHWFloatArray(Mat mat, int c, int h, int w) {
        // 从 Mat 中读取数据到 float[]
        float[] matData = new float[c * h * w];
        mat.get(0, 0, matData);  // 一次性读取所有数据

        float[][][] chwArray = new float[c][h][w];

        if (c == 1) {
            // 单通道
            int index = 0;
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    chwArray[0][row][col] = matData[index++];
                }
            }
        } else {
            // 多通道 (如 3 通道)
            // OpenCV 默认存储顺序是 HWC，但我们需要 CHW
            // matData 的排布是 row-major: b0 g0 r0 b1 g1 r1 ...
            // 这里假设通道顺序满足需求，否则需要手动再调换
            int index = 0;
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    for (int ch = 0; ch < c; ch++) {
                        // 对应于 Python 里的 transpose((2, 0, 1))
                        chwArray[ch][row][col] = matData[index++];
                    }
                }
            }
        }

        return chwArray;
    }

}
