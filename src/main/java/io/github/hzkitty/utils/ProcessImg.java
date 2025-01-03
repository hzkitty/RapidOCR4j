package io.github.hzkitty.utils;

import io.github.hzkitty.entity.Triple;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ProcessImg {

    // 缩小图像到最大边长 max_side_len
    public static Triple<Mat, Float, Float> reduceMaxSide(Mat img, int maxSideLen) throws ResizeImgError {
        int h = img.rows();
        int w = img.cols();

        float ratio = 1.0f;
        if (Math.max(h, w) > maxSideLen) {
            ratio = (h > w) ?  (float) maxSideLen / h :  (float) maxSideLen / w;
        }

        int resizeH = (int) Math.round(h * ratio / 32) * 32;
        int resizeW = (int) Math.round(w * ratio / 32) * 32;

        if (resizeW <= 0 || resizeH <= 0) {
            throw new ResizeImgError("resize_w 或 resize_h 小于等于 0");
        }

        Mat resizedImg = new Mat();
        Imgproc.resize(img, resizedImg, new Size(resizeW, resizeH));

        float ratioH = (float) h / resizeH;
        float ratioW = (float) w / resizeW;

        return Triple.of(resizedImg, ratioH, ratioW);
    }

    // 放大图像到最小边长 min_side_len
    public static Triple<Mat, Float, Float> increaseMinSide(Mat img, int minSideLen) throws ResizeImgError {
        int h = img.rows();
        int w = img.cols();

        float ratio = 1.0f;
        if (Math.min(h, w) < minSideLen) {
            ratio = (h < w) ? (float) minSideLen / h : (float) minSideLen / w;
        }

        int resizeH = (int) Math.round(h * ratio / 32) * 32;
        int resizeW = (int) Math.round(w * ratio / 32) * 32;

        if (resizeW <= 0 || resizeH <= 0) {
            throw new ResizeImgError("resize_w 或 resize_h 小于等于 0");
        }

        Mat resizedImg = new Mat();
        Imgproc.resize(img, resizedImg, new Size(resizeW, resizeH));

        float ratioH = (float) h / resizeH;
        float ratioW = (float) w / resizeW;

        return Triple.of(resizedImg, ratioH, ratioW);
    }

    // 添加带圆角边框的图像填充
    public static Mat addRoundLetterbox(Mat img, int top, int bottom, int left, int right) {
        Mat paddedImg = new Mat();
        Core.copyMakeBorder(img, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
        return paddedImg;
    }

    // 自定义异常类，用于捕获调整失败的情况
    public static class ResizeImgError extends Exception {
        public ResizeImgError(String message) {
            super(message);
        }
    }

}
