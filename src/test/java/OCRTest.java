import org.hzkitty.RapidOCR;
import org.hzkitty.entity.OcrResult;
import org.hzkitty.entity.ParamConfig;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;

public class OCRTest {

//    static {
//        nu.pattern.OpenCV.loadShared();
//    }

    @Test
    public void testPath() throws Exception {
        RapidOCR rapidOCR = RapidOCR.create();
        File file = new File("src/test/resources/text_01.png");
        String imgContent = file.getAbsolutePath();
        OcrResult ocrResult = rapidOCR.run(imgContent);
        Assertions.assertFalse(ocrResult.getRecRes().isEmpty());
        System.out.println(ocrResult);
    }

    @Test
    public void testBufferedImage() throws Exception {
        RapidOCR rapidOCR = RapidOCR.create();
        File file = new File("src/test/resources/text_01.png");
        BufferedImage imgContent = ImageIO.read(file);

        ParamConfig paramConfig = new ParamConfig();
        paramConfig.setReturnWordBox(true);
        OcrResult ocrResult = rapidOCR.run(imgContent, paramConfig);
        Assertions.assertFalse(ocrResult.getRecRes().isEmpty());
        System.out.println(ocrResult);
    }

    @Test
    public void testByte() throws Exception {
        RapidOCR rapidOCR = RapidOCR.create();
        File file = new File("src/test/resources/text_01.png");
        byte[] imgContent = Files.readAllBytes(file.toPath());
        OcrResult ocrResult = rapidOCR.run(imgContent);
        Assertions.assertFalse(ocrResult.getRecRes().isEmpty());
        System.out.println(ocrResult);
    }

    @Test
    public void testMat() throws Exception {
        RapidOCR rapidOCR = RapidOCR.create();
        File file = new File("src/test/resources/text_01.png");
        Mat imgContent = Imgcodecs.imread(file.getAbsolutePath());
        OcrResult ocrResult = rapidOCR.run(imgContent);
        Assertions.assertFalse(ocrResult.getRecRes().isEmpty());
        System.out.println(ocrResult);
    }

}
