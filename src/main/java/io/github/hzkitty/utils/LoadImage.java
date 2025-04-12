package io.github.hzkitty.utils;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 自定义异常类
 */
class LoadImageError extends Exception {
    public LoadImageError(String message) {
        super(message);
    }

    public LoadImageError(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * 将输入图片转换为 BGR 格式的 Mat
 */
public class LoadImage {

    /**
     * 核心方法
     *
     * @param img 可能是：
     *            1. String/Path: 图片路径
     *            2. byte[]: 图片文件的二进制数据
     *            3. Mat: 已经是OpenCV的Mat对象
     *            4. BufferedImage: 图片对象
     * @return OpenCV 的 Mat，通道顺序为 BGR
     * @throws LoadImageError 当加载或处理失败时，抛出异常
     */
    public Mat call(Object img) throws LoadImageError {
        if (!(img instanceof String || img instanceof Path || img instanceof byte[]
                || img instanceof Mat || img instanceof BufferedImage)) {
            throw new LoadImageError("输入类型不被支持: " + img.getClass().getName());
        }
        // 记录原始类型，用于后续逻辑判断
        Class<?> originImgType = img.getClass();

        Mat mat = loadImg(img);

        Mat converted = convertImg(mat, originImgType);

        return converted;
    }

    /**
     * 加载图片，转为 OpenCV 的 Mat 格式
     *
     * @param img 输入可能是多种类型
     * @return Mat
     * @throws LoadImageError 加载失败抛出异常
     */
    private Mat loadImg(Object img) throws LoadImageError {
        // 1. 如果是字符串或 Path，认为是图片文件路径
        if (img instanceof String || img instanceof Path) {
            String filePath = img instanceof String ? (String) img : ((Path) img).toString();
            verifyExist(filePath);
            boolean containsChinese = filePath.matches(".*[\\u4e00-\\u9fa5]+.*");
            Mat mat;
            if (!containsChinese) {
                mat = Imgcodecs.imread(filePath, Imgcodecs.IMREAD_COLOR);
            } else {
                // OpenCV 中的 imread 方法不支持中文路径，使用字节数组byte[]
                byte[] bytes;
                try {
                    bytes = Files.readAllBytes(Paths.get(filePath));
                } catch (IOException e) {
                    throw new LoadImageError("无法识别或读取图片: " + filePath);
                }
                MatOfByte mob = new MatOfByte(bytes);
                mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
            }
            if (mat.empty()) {
                throw new LoadImageError("无法识别或读取图片: " + filePath);
            }
            return mat;
        }

        // 2. 如果是 byte[]，认为是图片的二进制内容
        if (img instanceof byte[]) {
            byte[] bytes = (byte[]) img;
            MatOfByte mob = new MatOfByte(bytes);
            Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
            if (mat.empty()) {
                throw new LoadImageError("无法识别或读取二进制图片数据");
            }
            return mat;
        }

        // 3. 如果已经是 Mat，则直接返回
        if (img instanceof Mat) {
            return (Mat) img;
        }

        // 4. 如果已经是 Mat，则直接返回
        if (img instanceof BufferedImage) {
            // 可扩展：BufferedImage 转 Mat
            return bufferedImageToMat((BufferedImage) img);
        }

        // 4. 其他类型不支持
        throw new LoadImageError("不支持的图片输入类型: " + img.getClass().getName());
    }

    /**
     * 将图像转换为 BGR 三通道
     *
     * @param img           OpenCV 的 Mat（可能是多通道）
     * @param originImgType 原始输入类型，影响判断是否需要从 RGB 转 BGR
     * @return BGR 格式的 Mat
     * @throws LoadImageError 如果通道数/维度异常
     */
    private Mat convertImg(Mat img, Class<?> originImgType) throws LoadImageError {
        // OpenCV 的 Mat 通常是 2D（图像高度、宽度），通道数可以从 type 或者 shape 获得
        int channels = img.channels();
        int depth = img.depth();
        int rows = img.rows();
        int cols = img.cols();

        // 如果是单通道灰度
        if (channels == 1) {
            // 将灰度图转换为 BGR
            Mat bgrMat = new Mat();
            Imgproc.cvtColor(img, bgrMat, Imgproc.COLOR_GRAY2BGR);
            return bgrMat;
        }

        // 如果是两通道 (例如灰度 + alpha)
        if (channels == 2) {
            return cvtTwoToThree(img);
        }

        // 如果是三通道
        if (channels == 3) {
            if (String.class.isAssignableFrom(originImgType)
                    || Path.class.isAssignableFrom(originImgType)
                    || byte[].class.isAssignableFrom(originImgType)
                    || BufferedImage.class.isAssignableFrom(originImgType)) {
                // 假设原图是 RGB，需要转换成 BGR
                Mat bgrMat = new Mat();
                Imgproc.cvtColor(img, bgrMat, Imgproc.COLOR_RGB2BGR);
                return bgrMat;
            }
            // 如果原图已经是 BGR 格式，则直接返回
            return img;
        }

        // 如果是四通道 (例如 RGBA)
        if (channels == 4) {
            return cvtFourToThree(img);
        }

        // 如果通道数不在 [1, 2, 3, 4]，则抛出异常
        throw new LoadImageError("图像通道数(" + channels + ")不在[1, 2, 3, 4]范围内！");
    }

    /**
     * 将灰度+alpha 的两通道图转换为 BGR 三通道
     *
     * @param img 两通道图像
     * @return 转换后得到的 BGR
     */
    private Mat cvtTwoToThree(Mat img) {
        // 拆分通道: [灰度, alpha]
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

        // 第一个通道：灰度
        Mat gray = channels.get(0);
        // 第二个通道：alpha
        Mat alpha = channels.get(1);

        // 将灰度转换为 BGR
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);

        // alpha 取反（类似 bitwise_not）
        Mat notAlpha = new Mat();
        Core.bitwise_not(alpha, notAlpha);

        // 将 notAlpha 也转换为三通道，便于与 BGR 做后续操作
        Mat notAlpha3 = new Mat();
        Imgproc.cvtColor(notAlpha, notAlpha3, Imgproc.COLOR_GRAY2BGR);

        // 使用 alpha 作为掩膜，对原 BGR 部分进行保留
        Mat newImg = new Mat();
        Core.bitwise_and(bgr, bgr, newImg, alpha);

        // 将 newImg 与 notAlpha3 合并
        Core.add(newImg, notAlpha3, newImg);

        return newImg;
    }

    /**
     * 将 RGBA 的四通道图转换为 BGR 三通道
     *
     * @param img 四通道 RGBA
     * @return 转换后得到的 BGR
     */
    private Mat cvtFourToThree(Mat img) {
        // 拆分 RGBA
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels); // [R, G, B, A]

        Mat r = channels.get(0);
        Mat g = channels.get(1);
        Mat b = channels.get(2);
        Mat a = channels.get(3);

        // 合并 BGR (在 OpenCV 中通常顺序是 B, G, R)
        // 由于 Python 逻辑假定输入是 RGBA，因此这里顺序可做调整
        // 若原图确实是 RGBA，则 channels[0] = R, channels[1] = G, channels[2] = B
        // 这里想要得到 BGR 需要 merge(b, g, r)
        List<Mat> bgrList = new ArrayList<>();
        bgrList.add(b);
        bgrList.add(g);
        bgrList.add(r);
        Mat bgr = new Mat();
        Core.merge(bgrList, bgr);

        // alpha 取反
        Mat notA = new Mat();
        Core.bitwise_not(a, notA);

        // 转成三通道
        Mat notA3 = new Mat();
        Imgproc.cvtColor(notA, notA3, Imgproc.COLOR_GRAY2BGR);

        // 使用 alpha 作为掩膜，对 BGR 进行保留
        Mat newImg = new Mat();
        Core.bitwise_and(bgr, bgr, newImg, a);

        // 计算 newImg 的平均值（类似 np.mean(new_img)）
        // meanScalar.val[0], [1], [2], [3] 对应 B, G, R, alpha
        Scalar meanScalar = Core.mean(newImg);
        double avgColor = (meanScalar.val[0] + meanScalar.val[1] + meanScalar.val[2]) / 3.0;

        if (avgColor <= 0.0) {
            // 如果平均值接近 0，说明基本是黑色，则将 newImg 与 notA3 合并
            Core.add(newImg, notA3, newImg);
        } else {
            // 否则对 newImg 进行取反，得到类似黑色背景的效果
            Core.bitwise_not(newImg, newImg);
        }

        return newImg;
    }

    /**
     * 验证文件是否存在
     *
     * @param filePath 文件路径
     * @throws LoadImageError 不存在时抛出异常
     */
    private void verifyExist(String filePath) throws LoadImageError {
        File f = new File(filePath);
        if (!f.exists() || !f.isFile()) {
            throw new LoadImageError("文件不存在: " + filePath);
        }
    }

    /**
     * 将 BufferedImage 转为 Mat。这里的逻辑仅作示例，可根据实际业务调整。
     *
     * @param bi 传入的 BufferedImage
     * @return 转换后的 Mat
     */
    private Mat bufferedImageToMat(BufferedImage bi) {
        // 把 BufferedImage 当成三通道处理
        int width = bi.getWidth();
        int height = bi.getHeight();
        Mat mat = new Mat(height, width, CvType.CV_8UC3);

        // 将 pixel 拆分到 mat
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = bi.getRGB(x, y);
                // ARGB 格式
                int alpha = (pixel >> 24) & 0xFF;
                int red = (pixel >> 16) & 0xFF;
                int green = (pixel >> 8) & 0xFF;
                int blue = (pixel) & 0xFF;

                // 设置到 mat 中：OpenCV 默认顺序为 BGR
                mat.put(y, x, new byte[]{(byte) blue, (byte) green, (byte) red});
            }
        }
        // 如果图像是二值模式，转换为灰度图
        if (mat.channels() == 1) {
            Mat grayImg = new Mat();
            Imgproc.cvtColor(mat, grayImg, Imgproc.COLOR_BGR2GRAY);
            return grayImg;  // 返回灰度图
        }
        return mat;
    }

}

