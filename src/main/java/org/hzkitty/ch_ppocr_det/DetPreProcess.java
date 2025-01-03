package org.hzkitty.ch_ppocr_det;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * 图像检测前处理（使用 Scalar 来减均值、除标准差）
 */
public class DetPreProcess {
    // 均值、标准差、缩放因子
    private final Scalar meanScalar;
    private final Scalar stdScalar;
    private final double scale;

    // 限制边长与限制类型
    private final int limitSideLen;
    private final String limitType;

    /**
     * 构造函数，传入限制边长与限制类型（"max"或"min"），默认值可自行定义
     */
    public DetPreProcess(int limitSideLen, String limitType) {
        // 与 Python: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], scale=1/255.0
        this.meanScalar = new Scalar(0.5, 0.5, 0.5);
        this.stdScalar = new Scalar(0.5, 0.5, 0.5);
        this.scale = 1.0 / 255.0;

        this.limitSideLen = limitSideLen;
        // 若未指定则默认用 "min"
        this.limitType = (limitType == null ? "min" : limitType);
    }

    /**
     * 对图像进行预处理: 1) resize  2) normalize  3) permute  4) expandDims
     *
     * @param img OpenCV Mat 格式图像，通常为 BGR
     * @return 若成功，返回四维 float数组 [1, C, H, W]；若 resize 失败，返回 null
     */
    public float[][][][] call(Mat img) {
        // 1. resize
        Mat resizedImg = resize(img);
        if (resizedImg == null) {
            // 若 resize 后尺寸 <=0，返回 null
            return null;
        }

        // 2. normalize
        //   (pixel * scale - mean) / std，用 Scalar 与 Core API 实现
        Mat normImg = normalize(resizedImg);

        // 3. permute (H,W,C)->(C,H,W)
        float[][][] permuted = permute(normImg);

        // 4. expandDims => [1, C, H, W]
        return expandDims(permuted);
    }

    /**
     * resize 到 32 的倍数，
     * 若 limitType="max"，当 max(h,w)>limitSideLen 时才缩放；
     * 若 limitType="min"，当 min(h,w)<limitSideLen 时才放大；
     */
    private Mat resize(Mat img) {
        int h = img.rows();
        int w = img.cols();

        float ratio;
        if ("max".equalsIgnoreCase(limitType)) {
            // 若 max(h,w) > limitSideLen => ratio=limitSideLen/max(h,w) 否则 1.0
            if (Math.max(h, w) > limitSideLen) {
                ratio = (h > w) ? (float) limitSideLen / h : (float) limitSideLen / w;
            } else {
                ratio = 1.0f;
            }
        } else {
            // 若 min(h,w) < limitSideLen => ratio=limitSideLen/min(h,w) 否则 1.0
            if (Math.min(h, w) < limitSideLen) {
                ratio = (h < w) ? (float) limitSideLen / h : (float) limitSideLen / w;
            } else {
                ratio = 1.0f;
            }
        }

        int resizeH = Math.round(h * ratio);
        int resizeW = Math.round(w * ratio);

        // 调整到 32 的倍数
        resizeH = Math.round((float) Math.round(resizeH / 32.0) * 32);
        resizeW = Math.round((float) Math.round(resizeW / 32.0) * 32);

        if (resizeH <= 0 || resizeW <= 0) {
            return null;
        }

        // 使用 OpenCV 进行 resize
        Mat dst = new Mat();
        try {
            Imgproc.resize(img, dst, new Size(resizeW, resizeH));
        } catch (Exception e) {
            throw new ResizeImgError("Fail to resize image", e);
        }
        return dst;
    }

    /**
     * normalize：
     * 先将图像类型转为 CV_32FC3，并在 convertTo(...)时乘以 scale
     * 再用 Scalar 减去 mean，除以 std
     */
    private Mat normalize(Mat img) {
        // 先将 img 转成 float32，并乘以 scale => (img * scale)
        img.convertTo(img, CvType.CV_32FC3, scale);

        // 通过 Core.subtract & Core.divide 实现 (pixel - mean) / std
        // 现在 img = (img * scale)
        // subtract => (img - mean)
        Core.subtract(img, meanScalar, img);
        // divide => (img - mean) / std
        Core.divide(img, stdScalar, img);

        return img;
    }

    /**
     * 将图像从 (H,W,C) 转换为 (C,H,W) 的三维 float 数组
     */
    private float[][][] permute(Mat img) {
        int h = img.rows();
        int w = img.cols();
        int c = img.channels();
        float[][][] output = new float[c][h][w];

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                double[] pixel = img.get(row, col);  // size = c
                for (int ch = 0; ch < c; ch++) {
                    output[ch][row][col] = (float) pixel[ch];
                }
            }
        }
        return output;
    }

    /**
     * 扩展一个 batch 维度 => [1, C, H, W]
     */
    private float[][][][] expandDims(float[][][] permuted) {
        int c = permuted.length;
        int h = permuted[0].length;
        int w = permuted[0][0].length;

        float[][][][] out = new float[1][c][h][w];
        for (int cc = 0; cc < c; cc++) {
            for (int hh = 0; hh < h; hh++) {
                System.arraycopy(permuted[cc][hh], 0, out[0][cc][hh], 0, w);
            }
        }
        return out;
    }

    /**
     * 自定义异常，与 Python ResizeImgError 对应
     */
    public static class ResizeImgError extends RuntimeException {
        public ResizeImgError(String message, Throwable cause) {
            super(message, cause);
        }
    }

}
