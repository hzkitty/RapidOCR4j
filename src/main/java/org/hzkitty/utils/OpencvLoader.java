package org.hzkitty.utils;

import java.io.File;
import java.util.logging.Logger;

/**
 * Opencv环境依赖自动加载
 */
public class OpencvLoader {

    private final static Logger log = Logger.getLogger(OpencvLoader.class.getName());

    /**
     * 是否已经加载过原生库的标志位
     * 使用 volatile 修饰，保证线程可见性
     */
    private static volatile boolean loaded = false;

    /**
     * 用于多线程并发时的锁对象
     */
    private static final Object LOAD_LOCK = new Object();

    /**
     * 1. 根据指定文件名加载库（仅第一次生效，后续不会重复加载）
     */
    public static void loadOpencvLib(String filename) {
        // 如果已经加载过，直接返回
        if (loaded) {
            return;
        }

        // 先检查文件是否存在
        File libFile = new File(filename);
        if (!libFile.exists()) {
            throw new IllegalArgumentException("OpenCV native library file does not exist: " + filename);
        }

        synchronized (LOAD_LOCK) {
            // 再次判断，避免并发重复加载
            if (loaded) {
                return;
            }
            System.load(filename);
            loaded = true;
            log.info("Loaded OpenCV library success from " + filename);
        }
    }

    /**
     * 2. 自动识别当前系统并加载对应库（仅第一次生效，后续不会重复加载）
     */
    public static void loadOpencvLib() {
        // 如果已经加载过，直接返回
        if (loaded) {
            return;
        }

        synchronized (LOAD_LOCK) {
            // 再次判断，避免并发重复加载
            if (loaded) {
                return;
            }

            nu.pattern.OpenCV.loadShared();
            log.info("Loaded OpenCV library success");
        }
    }
}
