# RapidOCR4j

## 😺 项目介绍

- **本项目是多平台OCR工具，[RapidOCR](https://github.com/RapidAI/RapidOCR)的Java移植版本，采用ONNXRuntime作为推理引擎调用模型，包括使用OpenCV对图片的处理优化等**

> ✨如果该项目对您有帮助，您的star是我不断优化的动力！！！
>
> - [github点击前往](https://github.com/hzkitty/RapidOCR4j)
> - [gitee点击前往](https://gitee.com/hzkitty/RapidOCR4j)

## 👏 项目特点

- 纯Java代码调用ONNXRuntime + OpenCV，方便二次开发
- 支持CPU版本和GPU版本
- 支持传入Path、BufferedImage、byte[]、Mat
- 支持Windows、Linux、Mac平台，具体如下：

OS | Architecture
--- | ---
macOS | Intel
macOS | Apple Silicon (arm64)
Linux | x86_64
Linux | ARMv7 (arm)
Linux | ARMv8 (arm64 / aarch64)
Windows | x86_32
Windows | x86_64

目前跨平台主要是opencv的限制，如果是其他平台，可在本机手动编译opencv4.6.0，把平台二进制文件路径传给opencvLibPath参数
```java
OcrConfig ocrConfig = new OcrConfig();
ocrConfig.Global.setOpencvLibPath("src/test/resources/libopencv_java481.so");
RapidOCR rapidOCR = RapidOCR.create(ocrConfig);
```
## 🎉 快速开始

安装依赖，默认使用CPU版本
```xml
<dependency>
    <groupId>io.github.hzkitty</groupId>
    <artifactId>rapidocr4j</artifactId>
    <version>1.0.2</version>
</dependency>
```
使用示例
```java
RapidOCR rapidOCR = RapidOCR.create();
OcrResult ocrResult = rapidOCR.run("src/test/resources/text_01.png");
```
> ⚠️ 注意：1.0.0 版本，图片中文路径问题支持。见 [#2](https://github.com/hzkitty/RapidOCR4j/issues/2)

如果想要使用GPU, `onnxruntime_gpu` 对应版本可以在这里找到
[here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
```xml
<dependency>
    <groupId>io.github.hzkitty</groupId>
    <artifactId>rapidocr4j</artifactId>
    <version>1.0.2</version>
    <exclusions>
      <exclusion>
        <groupId>com.microsoft.onnxruntime</groupId>
        <artifactId>onnxruntime</artifactId>
      </exclusion>
    </exclusions>
</dependency>

<!-- 1.18.0 support CUDA 12.x -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.18.0</version>
</dependency>
```

[OcrConfig想更深入了解，请移步config.yaml参数解释](https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/)

>  安卓版本。使用 [RapidOcrAndroid4j](https://github.com/hzkitty/RapidOcrAndroid4j)


## 鸣谢

- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## 开源许可
使用 [Apache License 2.0](https://github.com/MyMonsterCat/DeviceTouch/blob/main/LICENSE)
