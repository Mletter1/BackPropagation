/**
 * Created by matthewletter on 10/7/14.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class NeuralNet
{
    private final int INPUT_NEURONS = 2;
    private final int NUMBER_OF_HIDDEN_LAYERS = 4;//TODO fix layers >4
    private final int OUTPUT_NEURONS = 4;
    private final double LEARNING_RATE = 0.02;    // Rho.
    private final int EPOCHES = 10;
    private final double SAMPLES = 800;

    public ArrayList<Layer> Layers = new ArrayList<Layer>();
    private double inputSampleValues[] = new double[INPUT_NEURONS];
    private double expected[] = new double[OUTPUT_NEURONS];

    private int classOutput[][] = new int[][]
            {{1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0},
                    {0, 0, 0, 1}};

    /**
     * entrypoint for network tests, sets up the network and starts running EPOCHES;
     */
    private void nNet()
    {
        int sample = 0;
        //class 1
        File f1 = new File("/Users/matthewletter/Documents/BackPropagation/data/TrainingData.txt");
        ArrayList<Sample> samples = parseFile(f1);
        printMinMaxOfData(samples);

        File f2 = new File("/Users/matthewletter/Documents/BackPropagation/data/TestingData.txt");
        ArrayList<Sample> test = parseFile(f2);
        printMinMaxOfData(test);
        buildLayers();

        System.out.println("\nbefore training");
        System.out.println("Network test is " + getRunStats(test) + "% correct.");

        double time = System.currentTimeMillis();

        // Train the network.
        for(int epoch = 0; epoch < EPOCHES; epoch++)
        {
            Collections.shuffle(samples);
            for(Sample s : samples) {
                inputSampleValues[0] = s.X1;
                inputSampleValues[1] = s.X2;
                sample = s.expectedClass;
                for (int i = 0; i < OUTPUT_NEURONS; i++) {
                    //System.out.println(sample + " : " + i);
                    expected[i] = classOutput[sample][i];
                }
                feedForward();
                backPropagation();
            }
            //System.out.println("Network test is " + getRunStats(test) + "% correct.");
        } // epoch

        System.out.println("\nfinished testing "+ EPOCHES + " epochs in "+((System.currentTimeMillis() - time)
                /1000)+" seconds");
        System.out.println("\nafter training");
        System.out.println("Network test is " + getRunStats(test) + "% correct.");

        //System.out.println("\nTest network against original input:");
        //testNetworkTraining(samples);

        //System.out.println("\nTest network against noisy input:");
        //testNetworkWithNoise1();
    }

    /**
     * build the layers of the network
     */
    private void buildLayers() {
        int previousLayerSize = INPUT_NEURONS;
        for (int i = 0; i < NUMBER_OF_HIDDEN_LAYERS; i++) {//add nodeOutputs layers
                Layers.add(new Layer(previousLayerSize,5));
            previousLayerSize = 5;
        }
        Layers.add(new Layer(previousLayerSize,4));
    }


    /**
     * feedforward implementation of the network
     */
    private void feedForward()
    {
        double sum = 0.0;
        // Calculate input to nodeOutputs layer.
        for(int hid = 0; hid < Layers.get(0).NUMBER_OF_NEURONS; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputSampleValues[inp] * Layers.get(0).inputToSelf[inp][hid];
            } // inp
            sum += Layers.get(0).inputToSelf[INPUT_NEURONS][hid];
            Layers.get(0).nodeOutputs[hid] = sigmoid(sum);
        } // hid
        //for each layer after the inputSampleValues
        for (int i = 1; i <= NUMBER_OF_HIDDEN_LAYERS; i++) {//<= to get the output layer
            // Calculate previousOutput to nodeOutput layer.
            for(int out = 0; out < Layers.get(i).NUMBER_OF_NEURONS; out++)
            {
                sum = 0.0;
                //iter over previous neurons outputs
                for(int nodeNum = 0; nodeNum < Layers.get(i-1).NUMBER_OF_NEURONS; nodeNum++)
                {
                    sum += Layers.get(i-1).nodeOutputs[nodeNum] * Layers.get(i)
                            .inputToSelf[nodeNum][out];
                }
                // Add in bias.
                sum += Layers.get(i).inputToSelf[INPUT_NEURONS][out];
                //set actual output to that of the sigmoid function
                Layers.get(i).nodeOutputs[out] = sigmoid(sum);
            }
        }
    }

    /**
     * runs the back propagation algorithm for neural nets
     */
    private void backPropagation()
    {
        // Calculate the output layer error
        for(int out = 0; out < Layers.get(Layers.size()-1).NUMBER_OF_NEURONS; out++)
        {
            Layers.get(Layers.size()-1).err[out] = (expected[out] - Layers.get(Layers.size()-1).nodeOutputs[out]) *
                    sigmoidDerivative(Layers.get(Layers.size()-1).nodeOutputs[out]);
            if((-1*(expected[out] - Layers.get(Layers.size()-1).nodeOutputs[out])) < .01){
                System.out.println(Layers.get(Layers.size()-1).err[out]);
            }else{
                System.out.println("good error:"+Layers.get(Layers.size()-1).err[out]);
            }
        }
        // Update the weights for the output layer.
        for(int out = 0; out < Layers.get(Layers.size()-1).NUMBER_OF_NEURONS; out++)
        {
            for(int hid = 0; hid < Layers.get(Layers.size()-2).NUMBER_OF_NEURONS; hid++)
            {
                Layers.get(Layers.size()-1).inputToSelf[hid][out] += (LEARNING_RATE * Layers.get(Layers.size()-1).err[out] *
                        Layers.get(Layers.size()-2).nodeOutputs[hid]);
            } // 1st hid
            Layers.get(Layers.size()-1).inputToSelf[Layers.get(Layers.size()-1).NUMBER_OF_NEURONS][out] +=
                    (LEARNING_RATE * Layers.get(Layers.size()-1).err[out]); // Update the bias.
        } // output

        // Calculate the nodeOutputs layer error
        for (int i = Layers.size()-1; i >0 ; i--) {
            // Calculate the nodeOutputs layer error (step 3 for nodeOutputs cell).
            for(int hid = 0; hid < Layers.get(i-1).NUMBER_OF_NEURONS; hid++)
            {
                Layers.get(i-1).err[hid] = 0.0;
                for(int out = 0; out < Layers.get(i).NUMBER_OF_NEURONS; out++)
                {
                    Layers.get(i-1).err[hid] += Layers.get(i).err[out] * Layers.get(i).inputToSelf[hid][out];
                }
                Layers.get(i-1).err[hid] *= sigmoidDerivative(Layers.get(i-1).nodeOutputs[hid]);
            }
            // Update the weights for the nodeOutputs layer (step 4).
            if(i-1 > 0) {
                for (int hid = 0; hid < Layers.get(i - 1).NUMBER_OF_NEURONS; hid++) {
                    for (int inp = 0; inp < Layers.get(i - 2).NUMBER_OF_NEURONS; inp++) {
                        Layers.get(i - 1).inputToSelf[inp][hid] += (LEARNING_RATE * Layers.get(i - 1).err[hid] * Layers.get(i
                                - 2).nodeOutputs[inp]);
                    }
                    Layers.get(i - 1).inputToSelf[Layers.get(i - 2).NUMBER_OF_NEURONS][hid] += (LEARNING_RATE * Layers.get(i
                            - 1).err[hid]); // Update the bias.
                }
            }
        }

        // Update the weights for the nodeOutputs layer (step 4).
        for(int hid = 0; hid < Layers.get(0).NUMBER_OF_NEURONS; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                Layers.get(0).inputToSelf[inp][hid] += (LEARNING_RATE * Layers.get(0).err[hid] * inputSampleValues[inp]);
            } // inp
            Layers.get(0).inputToSelf[INPUT_NEURONS][hid] += (LEARNING_RATE * Layers.get(0).err[hid]); // Update the bias.
        } // hid
        return;
    }

    /**
     * used to parse the provided text files
     * @param f file
     * @return ArrayList of Sample
     */
    public ArrayList<Sample> parseFile(File f){
        Scanner scanner;
        String[] sA;
        String s;
        ArrayList<Sample> samples = new ArrayList<Sample>();
        try {
            scanner = new Scanner(f);
            s = scanner.nextLine();
            s = s.replaceAll("\\s+"," ");
            //System.out.println(s);

            while(scanner.hasNext()){
                s = s.replaceAll("\\s+"," ");
                sA = s.split(" ");

                if(sA.length==4) {
                    samples.add(new Sample(Integer.parseInt(sA[0])-1,
                            Integer.parseInt(sA[0]), Double.parseDouble(sA[2]),
                            Double.parseDouble(sA[3])));
                }

                s = scanner.nextLine();

            }
            scanner.close();

        }catch(FileNotFoundException e){
            e.printStackTrace();
        }
        return samples;
    }

    /**
     * print the min and max samples as a double
     * @param samples list of x1 x2 values
     */
    private void printMinMaxOfData(ArrayList<Sample> samples){
        double max = 0;
        double min = 0;
        for(Sample s : samples){
            if(s.X1>max){
                max = s.X1;
            }
            if(s.X1<min){
                min = s.X1;
            }
            if(s.X2>max){
                max = s.X2;
            }
            if(s.X2<min){
                min = s.X2;
            }
        }
        System.out.println("min:"+min+" max:"+max);
    }

    /**
     * produce stats for a training epoche
     * @param samples input samples
     * @return
     */
    private double getRunStats(ArrayList<Sample> samples)
    {
        double sum = 0.0;
        for(Sample s : samples)
        {
            inputSampleValues[0] = s.X1;
            inputSampleValues[1] = s.X2;
            int sample = s.expectedClass;
            for(int j = 0; j < OUTPUT_NEURONS; j++)
            {
                expected[j] = classOutput[sample][j];
            }
            feedForward();
            if(max(Layers.get(Layers.size() - 1).nodeOutputs) == max(expected)){
                sum += 1;
            }
        }

        return (sum / SAMPLES * 100.0);
    }

    private void testNetworkTraining(ArrayList<Sample> samples)
    {
        // This function simply tests the training vectors against network.
        for(Sample s : samples)
        {
            inputSampleValues[0] = s.X1;
            inputSampleValues[1] = s.X2;

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(inputSampleValues[j] + "\t");
            } // j

            //System.out.print("Output: " + max(actual) + "\n");
        } // i

        return;
    }

    /**
     * as decribed in class, take the max output
     * @param outputVector
     * @return
     */
    private int max(double[] outputVector)
    {
        // This function returns the maxIndex of the max of outputVector().
        int maxIndex = 0;
        double max = outputVector[maxIndex];

        for(int i = 0; i < OUTPUT_NEURONS; i++)
        {
            if(outputVector[i] > max){
                max = outputVector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double sigmoid(double outputSum)
    {
        return (1.0 / (1.0 + Math.exp(-outputSum)));
    }

    private double sigmoidDerivative(double val)
    {
        return (val * (1.0 - val));
    }

    public static void main(String[] args)
    {
        new NeuralNet().nNet();
    }

}