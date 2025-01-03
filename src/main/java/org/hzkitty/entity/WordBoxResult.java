package org.hzkitty.entity;

import lombok.Data;
import org.opencv.core.Point;

import java.util.Arrays;
import java.util.List;

@Data
public class WordBoxResult {
    private List<String> wordBoxContentList;
    private List<Point[]> sortedWordBoxList;
    private List<Float> confList;

    public WordBoxResult(List<String> wordBoxContentList, List<Point[]> sortedWordBoxList, List<Float> confList) {
        this.wordBoxContentList = wordBoxContentList;
        this.sortedWordBoxList = sortedWordBoxList;
        this.confList = confList;
    }

    @Override
    public String toString() {
        return "WordBoxResult{" +
                "wordBoxContentList=" + wordBoxContentList +
                ", sortedWordBoxList=" + Arrays.deepToString(sortedWordBoxList.toArray()) +
                ", confList=" + confList +
                '}';
    }
}