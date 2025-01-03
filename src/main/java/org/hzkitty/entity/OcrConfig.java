package org.hzkitty.entity;

import lombok.*;

// OCR 主配置类
@Data
public class OcrConfig {
    public GlobalConfig Global = new GlobalConfig(); // 全局配置
    public DetConfig Det = new DetConfig(); // 检测模块配置
    public ClsConfig Cls = new ClsConfig(); // 分类模块配置
    public RecConfig Rec = new RecConfig(); // 识别模块配置


    // 全局配置类
    @Data
    public static class GlobalConfig {
        public float textScore = 0.5f; // 文本评分阈值
        public boolean useDet = true; // 是否使用检测模块
        public boolean useCls = true; // 是否使用分类模块
        public boolean useRec = true; // 是否使用识别模块
        public boolean printVerbose = false; // 是否打印详细信息
        public int minHeight = 30; // 最小高度
        public float widthHeightRatio = 8; // 宽高比
        public int maxSideLen = 2000; // 最大边长
        public int minSideLen = 30; // 最小边长
        public boolean returnWordBox = false; // 是否返回单词级别的框
        public int intraOpNumThreads = -1; // 单线程操作线程数
        public int interOpNumThreads = -1; // 多线程操作线程数

        public String opencvLibPath; // opencv环境依赖dll或so目录
    }

    // 检测模块配置类
    @Data
    public static class DetConfig {
        public int intraOpNumThreads = -1; // 单线程操作线程数
        public int interOpNumThreads = -1; // 多线程操作线程数
        public boolean useCuda = false; // 是否使用 CUDA
        public boolean useDml = false; // 是否使用 DML
        public String modelPath = "models/ch_PP-OCRv4_det_infer.onnx"; // 模型路径
        public int limitSideLen = 736; // 限制边长
        public String limitType = "min"; // 限制类型
        public float thresh = 0.3f; // 检测阈值
        public float boxThresh = 0.5f; // 边框阈值
        public int maxCandidates = 1000; // 最大候选框数
        public float unclipRatio = 1.6f; // 非极大值抑制后的扩展比例
        public boolean useDilation = true; // 是否使用膨胀操作
        public String scoreMode = "fast"; // 评分模式
        public boolean useArena = true; // arena内存池的扩展策略（速度有提升，但内存会剧增，且持续占用，不释放，默认开启）
    }

    // 分类模块配置类
    @Data
    public static class ClsConfig {
        public int intraOpNumThreads = -1; // 单线程操作线程数
        public int interOpNumThreads = -1; // 多线程操作线程数
        public boolean useCuda = false; // 是否使用 CUDA
        public boolean useDml = false; // 是否使用 DML
        public String modelPath = "models/ch_ppocr_mobile_v2.0_cls_infer.onnx"; // 模型路径
        public int[] clsImageShape = {3, 48, 192}; // 分类输入图像形状
        public int clsBatchNum = 1; // 分类批量处理数
        public float clsThresh = 0.9f; // 分类阈值
        public String[] labelList = {"0", "180"}; // 分类标签列表
        public boolean useArena = true; // arena内存池的扩展策略（速度有提升，但内存会剧增，且持续占用，不释放，默认开启）
    }

    // 识别模块配置类
    @Data
    public static class RecConfig {
        public int intraOpNumThreads = -1; // 单线程操作线程数
        public int interOpNumThreads = -1; // 多线程操作线程数
        public boolean useCuda = false; // 是否使用 CUDA
        public boolean useDml = false; // 是否使用 DML
        public String modelPath = "models/ch_PP-OCRv4_rec_infer.onnx"; // 模型路径
        public int[] recImgShape = {3, 48, 320}; // 识别输入图像形状
        public int recBatchNum = 1; // 识别批量处理数
        public boolean useArena = true; // arena内存池的扩展策略（速度有提升，但内存会剧增，且持续占用，不释放，默认开启）
        public String recKeysPath; // 字典路径，如果不设置，默认从模型获取
    }
}