package org.hzkitty.ch_ppocr_cls;

import org.hzkitty.entity.Pair;
import java.util.ArrayList;
import java.util.List;

public class ClsPostProcess {
    private String[] labelList;

    // 构造函数，初始化标签列表
    public ClsPostProcess(String[] labelList) {
        this.labelList = labelList;
    }

    // 调用方法，处理预测结果，返回标签和概率的配对列表
    public List<Pair<String, Float>> call(float[][] preds) {
        List<Pair<String, Float>> decodeOut = new ArrayList<>();
        for (int i = 0; i < preds.length; i++) {
            int idx = this.argMax(preds[i]);
            // 使用 ImmutablePair 创建不可变的键值对
            decodeOut.add(Pair.of(labelList[idx], preds[i][idx]));
        }
        return decodeOut;
    }

    // 辅助方法，找出数组中最大值的索引
    private int argMax(float[] array) {
        int bestIdx = 0;
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
