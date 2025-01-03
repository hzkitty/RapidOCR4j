package org.hzkitty.entity;

import lombok.Data;

@Data
public class OrtInferConfig {
    public int intraOpNumThreads; // 单线程操作线程数
    public int interOpNumThreads; // 多线程操作线程数
    public boolean useCuda; // 是否使用 CUDA
    public boolean useDml; // 是否使用 DML
    public String modelPath; // 模型路径
    public boolean useArena;
}
