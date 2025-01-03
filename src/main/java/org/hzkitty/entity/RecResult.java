package org.hzkitty.entity;

import lombok.Data;
import org.opencv.core.Point;

import java.util.Arrays;

@Data
public class RecResult {
    private Point[] dtBoxes;
    private final String text;
    private final float confidence;
    private WordBoxResult wordBoxResult;

    public RecResult(Point[] dtBoxes, String text, float confidence, WordBoxResult wordBoxResult) {
        this.dtBoxes = dtBoxes;
        this.text = text;
        this.confidence = confidence;
        this.wordBoxResult = wordBoxResult;
    }

    @Override
    public String toString() {
        return "RecResult{" +
                "dtBoxes=" + Arrays.toString(dtBoxes) +
                ", text='" + text + '\'' +
                ", confidence=" + confidence +
                ", wordBoxResult=" + wordBoxResult +
                '}';
    }
}
