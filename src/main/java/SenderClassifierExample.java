import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is a sender classifier that try uses the same hackthon data as run in Pytorch
 * to see if the same acc can be achieved via the same simple single layer nn but on java environment.
 *
 * Reference examples:
 * https://github.com/deeplearning4j/dl4j-examples/blob/7b4d76c9ff8de7697a1ff97ee917a10f5f3873e3/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java
 */
public class SenderClassifierExample {
    private static Logger log = LoggerFactory.getLogger(SenderClassifierExample.class);

    public static void main(String[] args) throws Exception {
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 15484;     //15485 values in each row of the feature_label_small.csv CSV: 15484 input features followed by an integer label (class) index. Labels are the 15485th value (index 15484) in each row
        int numClasses = 24501;     //24501 classes (types of senders) in the data set. Classes have integer values 0, 1 or 2 ... and so on
        int batchSize = 32;       //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        // 516348 examples, with batchSize is 64, around 16000 iterations per epoch
        int printIterationsNum = 200; // print score every 200 iterations

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("feature_label_train.csv").getFile()));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);

        RecordReader testRecordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        testRecordReader.initialize(new FileSplit(new ClassPathResource("feature_label_test.csv").getFile()));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader,batchSize,labelIndex,numClasses);

//        allData.shuffle();
//        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training
//
//        DataSet trainingData = testAndTrain.getTrain();
//        DataSet testData = testAndTrain.getTest();

        final int numInputs = 15484;
        int hiddenLayer1Num = 2000;
        int iterations = 1;
        long seed = 42;
        int nEpochs = 20;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .iterations(iterations)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.02)
                .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
//                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(hiddenLayer1Num)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(hiddenLayer1Num).nOut(numClasses).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(printIterationsNum));

        for ( int n = 0; n < nEpochs; n++) {

            model.fit(trainIter);

            // evaluate the model on test data, once every epoch
            if ((n + 1) % 1 == 0) {
                //evaluate the model on the test set
                Evaluation eval = new Evaluation(numClasses);

                testIter.reset();

                int i = 0;
                while(testIter.hasNext()) {
                    DataSet t = testIter.next();
                    INDArray features = t.getFeatures();
                    INDArray labels = t.getLabels();
                    INDArray predicted = model.output(features, false);

                    eval.eval(labels, predicted);

                    if (i * batchSize > 10000) break; // When number of testing data is too high, we have out of memory issue on GPU.
                    // TODO A CUDA error occurs when labels' lengths is above 15000
                    // TODO Test this issue without GPU and make a sample code example later to see if it repeat

                    i ++;
                }

                log.info(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                        n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
                // log.info(eval.stats());
            }
        }

        System.out.println("Finished...");
    }
}
