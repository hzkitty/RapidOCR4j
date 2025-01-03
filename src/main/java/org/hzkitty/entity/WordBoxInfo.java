package org.hzkitty.entity;

import lombok.Data;

import java.util.List;

@Data
public class WordBoxInfo {
    private Double textIndexLen;
    private List<List<String>> wordList;
    private List<List<Integer>> wordColList;
    private List<String> stateList;
    private List<Float> confList;

    public WordBoxInfo(Double textIndexLen, List<List<String>> wordList, List<List<Integer>> wordColList, List<String> stateList, List<Float> confList) {
        this.textIndexLen = textIndexLen;
        this.wordList = wordList;
        this.wordColList = wordColList;
        this.stateList = stateList;
        this.confList = confList;
    }

    @Override
    public String toString() {
        return "WordBoxInfo{" +
                "textIndexLen=" + textIndexLen +
                ", wordList=" + wordList +
                ", wordColList=" + wordColList +
                ", stateList=" + stateList +
                ", confList=" + confList +
                '}';
    }
}