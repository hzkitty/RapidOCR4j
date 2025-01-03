package io.github.hzkitty.entity;

import lombok.Data;

@Data
public class ParamConfig {
    public Float boxThresh; // 边框阈值
    public Float unclipRatio; // 非极大值抑制后的扩展比例
    public Float textScore; // 文本评分阈值
    public Boolean returnWordBox; // 是否返回单词级别的框
    public Boolean useDet; // 是否使用检测模块
    public Boolean useCls; // 是否使用分类模块
    public Boolean useRec; // 是否使用识别模块
}
