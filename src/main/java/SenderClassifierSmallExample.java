import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class SenderClassifierSmallExample {
    private static Logger log = LoggerFactory.getLogger(SenderClassifierSmallExample.class);

    public static void main(String[] args) throws Exception {
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 15484;     //15485 values in each row of the feature_label_small.csv CSV: 15484 input features followed by an integer label (class) index. Labels are the 15485th value (index 15484) in each row
        int numClasses = 24501;     //24501 classes (types of senders) in the data set. Classes have integer values 0, 1 or 2 ... and so on
        int batchSize = 2;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("feature_label_small.csv").getFile()));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);

        RecordReader testRecordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        testRecordReader.initialize(new FileSplit(new ClassPathResource("feature_label_small.csv").getFile()));
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
        int nEpochs = 50;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.02)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(hiddenLayer1Num)
                        .build())
//                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
//                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(hiddenLayer1Num).nOut(numClasses).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for ( int n = 0; n < nEpochs; n++) {
            model.fit(trainIter);

            // evaluate the model once every 10th epoch
            if ((n + 1) % 10 == 0) {
                //evaluate the model on the test set
                Evaluation eval = new Evaluation(numClasses);

                testIter.reset();
                DataSet t = testIter.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray predicted = model.output(features, false);

                while(testIter.hasNext()) {
                    DataSet t2 = testIter.next();
                    INDArray features2 = t2.getFeatures();
                    INDArray labels2 = t2.getLabels();
                    INDArray predicted2 = model.output(features2, false);

                    labels = Nd4j.vstack(labels,labels2);
                    predicted = Nd4j.vstack(predicted,predicted2);
                }

                eval.eval(labels, predicted);
                log.info(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f] ",
                        n +1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()
                ));
                // log.info(eval.stats());
            }
        }

        System.out.println("Finished...");
    }
}
