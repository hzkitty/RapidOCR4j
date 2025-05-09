package io.github.hzkitty.utils;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import io.github.hzkitty.entity.OrtInferConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

public class OrtInferSession {

    private static final Logger logger = LoggerFactory.getLogger(OrtInferSession.class);

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    private boolean useCuda = false;
    private boolean useDirectML = false;


    public OrtInferSession(OrtInferConfig ortInferConfig) {
        logger.info("Initializing OrtInferSession...");

        String modelPath = ortInferConfig.getModelPath();
        this.useCuda = ortInferConfig.useCuda;
        this.useDirectML = ortInferConfig.useDml;

        // 1、创建 ONNX Runtime 环境
        this.env = OrtEnvironment.getEnvironment("OrtInferSessionEnv");

        try {
            // 2. 初始化 SessionOptions
            SessionOptions sessionOptions = initSessionOptions(ortInferConfig);
            EnumSet<OrtProvider> availableProviders = env.getAvailableProviders();
            if (this.useCuda && availableProviders.contains(OrtProvider.CUDA)) {
                OrtCUDAProviderOptions providerOptions = new OrtCUDAProviderOptions(ortInferConfig.getDeviceId());
                // kNextPowerOfTwo（默认值）以 2 的幂数扩展，而 kSameAsRequested 每次扩展的大小与分配请求的大小相同。
                providerOptions.add("arena_extend_strategy", "kNextPowerOfTwo");
                providerOptions.add("cudnn_conv_algo_search", "EXHAUSTIVE");
                providerOptions.add("do_copy_in_default_stream", "1");
                sessionOptions.addCUDA(providerOptions);
                logger.info("Requested CUDA EP added to session options, deviceId: {}.", ortInferConfig.getDeviceId());
            }

            if (this.useDirectML && availableProviders.contains(OrtProvider.DIRECT_ML)) {
                sessionOptions.addDirectML(ortInferConfig.getDeviceId());
                logger.info("Requested DirectML EP - might not be supported in certain Java packages.");
            }

            // 最后添加 CPU
            if (availableProviders.contains(OrtProvider.CPU)) {
                sessionOptions.addCPU(ortInferConfig.useArena);
                logger.info("CPU EP added to session options.");
            }

            // 4. 创建推理会话
            Path path = Paths.get(modelPath);
            if (path.isAbsolute()) {
                if (!path.toFile().exists()) {
                    throw new RuntimeException("模型文件未找到: " + modelPath);
                }
                this.session = env.createSession(modelPath, sessionOptions);
            } else {
                InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream(modelPath);
                this.session = env.createSession(loadModel(inputStream), sessionOptions);
            }
            inputName = this.getInputNames().get(0);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        logger.info("OrtInferSession initialization completed.");
    }

    /**
     * 初始化 SessionOptions
     */
    private SessionOptions initSessionOptions(OrtInferConfig ortInferConfig) throws OrtException {
        SessionOptions sessOpt = new SessionOptions();
        int cpuNums = Runtime.getRuntime().availableProcessors();

        int intraOpNumThreads = ortInferConfig.intraOpNumThreads;
        if (intraOpNumThreads >= 1 && intraOpNumThreads <= cpuNums) {
            sessOpt.setIntraOpNumThreads(intraOpNumThreads);
        }
        int interOpNumThreads = ortInferConfig.interOpNumThreads;
        if (interOpNumThreads >= 1 && interOpNumThreads <= cpuNums) {
            sessOpt.setInterOpNumThreads(interOpNumThreads);
        }
        // 禁用 arena 内存池的扩展策略
        sessOpt.setCPUArenaAllocator(ortInferConfig.useArena);
        // 启用图优化
        sessOpt.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        // 日志等级
        sessOpt.setSessionLogVerbosityLevel(4);
        sessOpt.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL);
        return sessOpt;
    }

    /**
     * 执行推理
     *
     * @param inputData 传入的输入张量数据(形状需根据模型而定)
     * @return 推理结果 (形状需根据模型而定)
     */
    public Object run(float[][][][] inputData) throws OrtException {
        int dim1 = inputData.length;
        int dim2 = inputData[0].length;
        int dim3 = inputData[0][0].length;
        int dim4 = inputData[0][0][0].length;

        // 一维数组的大小
        int totalSize = dim1 * dim2 * dim3 * dim4;

        // 使用流将多维数组转为一维数组
        float[] dataArray = new float[totalSize];
        int index = 0;
        for (float[][][] item1 : inputData) {
            for (float[][] item2 : item1) {
                for (float[] item3 : item2) {
                    System.arraycopy(item3, 0, dataArray, index, item3.length);
                    index += item3.length;
                }
            }
        }

        long[] shape = new long[]{dim1, dim2, dim3, dim4};
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(dataArray), shape)) {
            try (Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                // 获取输出张量的值
                OnnxTensor onnxValue = (OnnxTensor) result.get(0);
                return onnxValue.getValue();
            }
        }
    }

    /**
     * 获取模型的输入名字列表
     */
    public List<String> getInputNames() throws OrtException {
        List<String> names = new ArrayList<>();
        for (NodeInfo info : session.getInputInfo().values()) {
            names.add(info.getName());
        }
        return names;
    }

    /**
     * 获取模型的输出名字列表
     */
    public List<String> getOutputNames() throws OrtException {
        List<String> names = new ArrayList<>();
        for (NodeInfo info : session.getOutputInfo().values()) {
            names.add(info.getName());
        }
        return names;
    }

    /**
     * 在 ONNX 模型的 metadata 中提取指定 key 对应的数据，split line 后返回
     */
    public List<String> getCharacterList(String key) {
        try {
            Map<String, String> customMetadata = session.getMetadata().getCustomMetadata();
            if (customMetadata.containsKey(key)) {
                String content = customMetadata.get(key);
                // 按行拆分
                return Arrays.asList(content.split("\\r?\\n"));
            }
            return Collections.emptyList();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 判断模型中是否包含指定 key
     */
    public boolean haveKey(String key) {
        try {
            Map<String, String> customMetadata = session.getMetadata().getCustomMetadata();
            return customMetadata.containsKey(key);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    private byte[] loadModel(InputStream modelInputStream) {
        try (
                InputStream inputStream = modelInputStream;
                ByteArrayOutputStream buffer = new ByteArrayOutputStream()
        ) {
            int nRead;
            byte[] data = new byte[1024];

            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }

            buffer.flush();
            return buffer.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
