package io.github.hzkitty.ch_ppocr_rec;

import ai.onnxruntime.OrtException;
import io.github.hzkitty.entity.OcrConfig;
import io.github.hzkitty.entity.OrtInferConfig;
import io.github.hzkitty.entity.Pair;
import io.github.hzkitty.entity.TupleResult;
import io.github.hzkitty.utils.OrtInferSession;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.stream.IntStream;

public class TextRecognizer {

    private final OrtInferSession session;
    private final CTCLabelDecode postprocessOp;

    private final int recBatchNum;         // 识别批处理大小
    private final int[] recImageShape;     // 识别输入形状 (如 [3, 32, 320])

    /**
     * 构造方法
     *
     * @param recConfig 配置字典，包含模型和识别参数
     */
    public TextRecognizer(OcrConfig.RecConfig recConfig) {
        // 初始化推理会话
        OrtInferConfig ortInferConfig = new OrtInferConfig();
        ortInferConfig.setIntraOpNumThreads(recConfig.getIntraOpNumThreads());
        ortInferConfig.setInterOpNumThreads(recConfig.getInterOpNumThreads());
        ortInferConfig.setUseCuda(recConfig.isUseArena());
        ortInferConfig.setUseDml(recConfig.isUseDml());
        ortInferConfig.setModelPath(recConfig.getModelPath());
        ortInferConfig.setUseArena(recConfig.isUseArena());
        // 1. 创建 ONNX 推理会话
        this.session = new OrtInferSession(ortInferConfig);

        // 2. 判断是否可从会话内获取字符表
        List<String> character = null;
        if (this.session.haveKey("character")) {
            character = this.session.getCharacterList("character");
        }

        // 3. 获取自定义字符文件路径
        String characterPath = recConfig.getRecKeysPath();

        // 4. 初始化 CTC 后处理类
        this.postprocessOp = new CTCLabelDecode(character, characterPath);

        // 5. 批处理数量 & 图像形状
        this.recBatchNum = recConfig.getRecBatchNum();
        // 例如 [3, 32, 320]
        this.recImageShape = recConfig.getRecImgShape();
    }

    /**
     * 进行文本识别
     *
     * @param imgList       多张图像的列表
     * @param returnWordBox 是否返回单词边界信息
     * @return (识别结果列表, 总耗时)
     */
    public Pair<List<TupleResult>, Double> call(List<Mat> imgList, boolean returnWordBox) throws OrtException {
        // 若只有单张图，可封装成 List
        if (imgList == null || imgList.isEmpty()) {
            return Pair.of(Collections.emptyList(), 0.0);
        }

        // 1. 计算每张图像的宽高比
        double[] widthArray = new double[imgList.size()];
        for (int i = 0; i < imgList.size(); i++) {
            Mat img = imgList.get(i);
            double w = img.width() * 1.0;
            double h = img.height() * 1.0;
            widthArray[i] = w / h;
        }

        // 2. 对宽高比排序，返回索引
        Integer[] indices = IntStream.range(0, widthArray.length)
                .boxed()
                .toArray(Integer[]::new);
        Arrays.sort(indices, Comparator.comparingDouble(o -> widthArray[o]));

        // 3. 准备结果数组，用来存放每张图像的识别结果
        TupleResult[] recRes = new TupleResult[imgList.size()];
        // 先初始化空值
        for (int i = 0; i < recRes.length; i++) {
            recRes[i] = new TupleResult("", 0.0f, null);
        }

        // 4. 批量处理
        int imgNum = imgList.size();
        double totalElapse = 0.0;

        for (int beg = 0; beg < imgNum; beg += recBatchNum) {
            int end = Math.min(imgNum, beg + recBatchNum);

            // 获取网络输入形状(如 3, 32, 320)
            int imgC = recImageShape[0];
            int imgH = recImageShape[1];
            int imgW = recImageShape[2];

            // 在批次内部，统计最大宽高比
            float maxWhRatio = (float) imgW / imgH;
            List<Float> whRatioList = new ArrayList<>();

            for (int i = beg; i < end; i++) {
                int idx = indices[i];
                Mat curImg = imgList.get(idx);
                float curH = curImg.height();
                float curW = curImg.width();
                float whRatio = curW / curH;
                if (whRatio > maxWhRatio) {
                    maxWhRatio = whRatio;
                }
                whRatioList.add(whRatio);
            }

            // 5. 归一化 & 调整图像大小
            List<float[][][]> normImgBatchList = new ArrayList<>();
            for (int i = beg; i < end; i++) {
                int idx = indices[i];
                Mat curImg = imgList.get(idx);
                float[][][] normImg = resizeNormImg(curImg, maxWhRatio);
                normImgBatchList.add(normImg);
            }

            // 将批次图像拼接成 [batchSize, channels, height, width]
            int batchSize = normImgBatchList.size();
            float[][][][] normImgBatch = new float[batchSize][imgC][imgH][];
            for (int bIndex = 0; bIndex < batchSize; bIndex++) {
                normImgBatch[bIndex] = normImgBatchList.get(bIndex);
            }

            // 6. 推理
            long startTime = System.currentTimeMillis();
            float[][][] preds = (float[][][]) session.run(normImgBatch);
            double elapseSec = (System.currentTimeMillis() - startTime) / 1000.0;
            totalElapse += elapseSec;

            // 7. 后处理
            List<DecodeResult> recResult = postprocessOp.call(preds, returnWordBox, whRatioList, maxWhRatio);

            // 8. 将结果放回到对应的下标
            for (int rno = 0; rno < recResult.size(); rno++) {
                DecodeResult dr = recResult.get(rno);
                int origIdx = indices[beg + rno];
                recRes[origIdx] = new TupleResult(dr.getText(), dr.getConfidence(), dr.getWordBoxInfo());
            }
        }

        // 返回 (识别结果, 总耗时)
        List<TupleResult> resultList = Arrays.asList(recRes);
        return Pair.of(resultList, totalElapse);
    }

    /**
     * 调整和归一化图像。
     *
     * @param imgMat     OpenCV Mat 格式的图像
     * @param maxWhRatio 最大宽高比
     * @return 归一化后的 float 三维数组，格式为 [通道数][高度][宽度]
     */
    public float[][][] resizeNormImg(Mat imgMat, float maxWhRatio) {
        int imgChannel = recImageShape[0];
        int imgHeight = recImageShape[1];
//        int imgWidth = recImageShape[2];

        // 确保输入图像的通道数与预期一致
        if (imgMat.channels() != imgChannel) {
            throw new IllegalArgumentException("输入图像的通道数与预期不符");
        }

        // 计算新的图像宽度
        int imgWidth = (int) (imgHeight * maxWhRatio);

        // 获取原图的高度和宽度
        int originalHeight = imgMat.rows();
        int originalWidth = imgMat.cols();

        // 计算宽高比
        float ratio = (float) originalWidth / (float) originalHeight;

        // 判断调整后的宽度是否超过限制
        int resizedW;
        if (Math.ceil(imgHeight * ratio) > imgWidth) {
            resizedW = imgWidth;
        } else {
            resizedW = (int) Math.ceil(imgHeight * ratio);
        }

        // 调整图像大小
        Size size = new Size(resizedW, imgHeight);
        Mat resizedMat = new Mat();
        Imgproc.resize(imgMat, resizedMat, size);

        // 将图像转换为浮点类型
        Mat resizedFloatMat = new Mat();
        resizedMat.convertTo(resizedFloatMat, CvType.CV_32FC(imgChannel));

        // 初始化填充后的三维数组，默认值为 0
        float[][][] paddingIm = new float[imgChannel][imgHeight][imgWidth];
        for (int c = 0; c < imgChannel; c++) {
            for (int i = 0; i < imgHeight; i++) {
                for (int j = 0; j < imgWidth; j++) {
                    paddingIm[c][i][j] = 0.0f;
                }
            }
        }

        // 遍历调整后的图像，并进行归一化处理
        for (int i = 0; i < imgHeight; i++) {
            for (int j = 0; j < resizedW; j++) {
                // 获取当前像素的所有通道值
                float[] pixel = new float[imgChannel];
                resizedFloatMat.get(i, j, pixel);
                for (int c = 0; c < imgChannel; c++) {
                    // 归一化: (value / 255.0 - 0.5) / 0.5 = value / 255.0 * 2 - 1.0
                    float normalizedValue = (pixel[c] / 255.0f) * 2.0f - 1.0f;
                    paddingIm[c][i][j] = normalizedValue;
                }
            }
        }

        return paddingIm;
    }

}
