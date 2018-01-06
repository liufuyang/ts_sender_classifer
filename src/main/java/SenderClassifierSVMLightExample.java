import org.datavec.api.conf.Configuration;
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
import recordreader.MySVMLightRecordReader;

/**
 * This is a sender classifier that try uses the same hackthon data as run in Pytorch
 * to see if the same acc can be achieved via the same simple single layer nn but on java environment.
 *
 * Reference examples:
 * https://github.com/deeplearning4j/dl4j-examples/blob/7b4d76c9ff8de7697a1ff97ee917a10f5f3873e3/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java
 */
public class SenderClassifierSVMLightExample {
    private static Logger log = LoggerFactory.getLogger(SenderClassifierSVMLightExample.class);

    public static void main(String[] args) throws Exception {

        int labelIndex = 15484;         //15485 values in each row of the feature_label_small.csv CSV: 15484 input features followed by an integer label (class) index. Labels are the 15485th value (index 15484) in each row
        int numClasses = 24501;         //24501 classes (types of senders) in the data set. Classes have integer values 0, 1 or 2 ... and so on
        int batchSize = 8;              // 516348 examples, with batchSize is 8, around 64000 iterations per epoch
        int printIterationsNum = 8000;  // print score every 8000 iterations

        Configuration config = new Configuration();
        config.setBoolean(MySVMLightRecordReader.ZERO_BASED_INDEXING, true);
        config.setInt(MySVMLightRecordReader.NUM_FEATURES, labelIndex);

        MySVMLightRecordReader trainRecordReader = new MySVMLightRecordReader();
        trainRecordReader.initialize(config, new FileSplit(new ClassPathResource("feature_label.svmlight.train.txt").getFile()));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader,batchSize,labelIndex,numClasses);

        MySVMLightRecordReader testRecordReader = new MySVMLightRecordReader();
        testRecordReader.initialize(config, new FileSplit(new ClassPathResource("feature_label.svmlight.test.txt").getFile()));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader,batchSize,labelIndex,numClasses);

        final int numInputs = 15484;
        int hiddenLayer1Num = 2000;
        int iterations = 1;
        long seed = 42;
        int nEpochs = 10;

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

//                    if (i * batchSize > 10000) break; // When number of testing data is too high, we have out of memory issue on GPU.
                    // TODO A CUDA error occurs when labels' lengths is above 15000
                    // TODO Test this issue without GPU and make a sample code example later to see if it repeat

                    i ++;
                }

                log.info(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                        n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
                // log.info(eval.stats());
            }
        }
        log.info("Finished...");
    }
}
