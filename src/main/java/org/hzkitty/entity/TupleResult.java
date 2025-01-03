package org.hzkitty.entity;

import lombok.Data;

/**
 * 用于存储文本识别结果 (text, conf)
 */
@Data
public class TupleResult {
    private final String text;
    private final float confidence;
    private WordBoxInfo wordBoxInfo;
    private WordBoxResult wordBoxResult;

    public TupleResult(String text, float confidence, WordBoxInfo wordBoxInfo) {
        this.text = text;
        this.confidence = confidence;
        this.wordBoxInfo = wordBoxInfo;
    }

    @Override
    public String toString() {
        return "TupleResult{" +
                ", text='" + text + '\'' +
                ", confidence=" + confidence +
                ", wordBoxInfo=" + wordBoxInfo +
                ", wordBoxResult=" + wordBoxResult +
                '}';
    }
}
