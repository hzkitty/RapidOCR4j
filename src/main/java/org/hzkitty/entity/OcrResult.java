package org.hzkitty.entity;

import lombok.Data;

import java.util.List;

/**
 * OCR 识别结果
 */
@Data
public class OcrResult {
    private String strRes;
    private List<RecResult> recRes;
    private double elapseTime;  // 总耗时
    private double detTime;  // 检测耗时
    private double clsTime;  // 分类耗时
    private double recTime;  // 识别耗时

    public OcrResult(String strRes, List<RecResult> recRes, double elapseTime, double detTime, double clsTime, double recTime) {
        this.strRes = strRes;
        this.recRes = recRes;
        this.elapseTime = elapseTime;
        this.detTime = detTime;
        this.clsTime = clsTime;
        this.recTime = recTime;
    }

    @Override
    public String toString() {
        return "OcrResult{" +
                "strRes='" + strRes + '\'' +
                ", recRes=" + recRes +
                ", elapseTime=" + elapseTime +
                ", detTime=" + detTime +
                ", clsTime=" + clsTime +
                ", recTime=" + recTime +
                '}';
    }
}