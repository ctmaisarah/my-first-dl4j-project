package ai.certifai.exdark;

import org.apache.commons.io.FileUtils;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.util.ArchiveUtils;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class ExDarkDataSetIterator_OldData {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ExDarkDataSetIterator_OldData.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static String downloadLink;
    private static Path trainDir, testDir;
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, Path dir, int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(dir.toString()));
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, testDir, batchSize);
    }

    public static void setup() throws IOException {
        log.info("Load data...");
        loadData();
        trainDir = Paths.get(dataDir, "ExDark_OldData/Train");
        testDir = Paths.get(dataDir, "ExDark_OldData/Test");
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
    }

    private static void loadData() throws IOException {
        dataDir = "src/main/resources";
        downloadLink = "https://www.googleapis.com/drive/v3/files/1AnUDAMLRmFzpCDhG1gha048y5wmm9CkZ?alt=media&key=AIzaSyB8YabIWgSWZYLO6Mb0F7d3Mxc9omqC2XE";
        File parentDir = new File(Paths.get(dataDir, "ExDark_OldData").toString());
        if (!parentDir.exists()) {
            downloadAndUnzip();
        }

    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = (new File(dataDir)).getAbsolutePath();
        File zipFile = new File(dataPath, "ExDark_OldData.zip");
        if (!zipFile.isFile()) {
            log.info("Downloading the dataset from " + downloadLink + "...");
            FileUtils.copyURLToFile(new URL(downloadLink), zipFile);
            log.info("Downloaded file is complete");
        }

        log.info("Unzipping " + zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }
}


