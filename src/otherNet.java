/**
 * Created by matthewletter on 10/7/14.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.text.DecimalFormat;
import java.util.Scanner;

public class otherNet
{

    private final int INPUT_NEURONS = 2;
    public int numberOfHiddenNeuronsPerLayer = 2;
    public int numebrOfHiddenLayers = 1;
    private final int OUTPUT_NEURONS = 4;
    public double bias = 1;

    public double lRate = 0.02;    // Rho.
    public int epoches = 100;

    // Input to Hidden Weights (with Biases).
    private double[][] inputToHiddenWeights = new double[INPUT_NEURONS + 1][numberOfHiddenNeuronsPerLayer];

    private double[][][] hiddenLayers = new double[numebrOfHiddenLayers][numberOfHiddenNeuronsPerLayer +
            1][numberOfHiddenNeuronsPerLayer];

    // Hidden to Output Weights (with Biases).
    private double[][] hiddenToOutputWeights = new double[numberOfHiddenNeuronsPerLayer + 1][OUTPUT_NEURONS];


    // Activations.
    private double[] inputs = new double[INPUT_NEURONS];
    private double[] hidden = new double[numberOfHiddenNeuronsPerLayer];
    public double[][] hiddenOutputs=new double[numebrOfHiddenLayers][numberOfHiddenNeuronsPerLayer];
    /* TODO update hidden outputs to use the above*/

    private double[] target = new double[OUTPUT_NEURONS];
    private double[] actual = new double[OUTPUT_NEURONS];

    // Unit errors.
    private double erro[] = new double[OUTPUT_NEURONS];
    private double errh[] = new double[numberOfHiddenNeuronsPerLayer];

    private final int MAX_SAMPLES = 800;

    private int trainInputs[][] = new int[][] {{1, 1, 1, 0},
            {1, 1, 0, 0},
            {0, 1, 1, 0},
            {1, 0, 1, 0},
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {1, 1, 1, 1},
            {1, 1, 0, 1},
            {0, 1, 1, 1},
            {1, 0, 1, 1},
            {1, 0, 0, 1},
            {0, 1, 0, 1},
            {0, 0, 1, 1}};

    private int trainOutput[][] = new int[][]
            {{1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0},
                    {0, 0, 0, 1}};

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
    private void NeuralNetwork()
    {
        if(numebrOfHiddenLayers>1){
            hiddenLayers = new double[numebrOfHiddenLayers][numberOfHiddenNeuronsPerLayer +
                    1][numberOfHiddenNeuronsPerLayer];
            hiddenOutputs = new double[numebrOfHiddenLayers][numberOfHiddenNeuronsPerLayer];
        }
        int sample = 0;
        //class 1
        File f1 = new File("/Users/matthewletter/Documents/BackPropagation/data/TrainingData.txt");
        ArrayList<Sample> samples = parseFile(f1);
        printMinMaxOfData(samples);

        File f2 = new File("/Users/matthewletter/Documents/BackPropagation/data/TestingData.txt");
        ArrayList<Sample> test = parseFile(f2);
        printMinMaxOfData(test);

        assignRandomWeights();
        System.out.println("\nbefore training");
        getTrainingStats(samples);

        double time = System.currentTimeMillis();

        // Train the network.
        for(int epoch = 0; epoch < epoches; epoch++)
        {
            Collections.shuffle(samples);
            for(Sample s : samples) {
                inputs[0] = s.X1;
                inputs[1] = s.X2;
                sample = s.expectedClass;
                for (int i = 0; i < OUTPUT_NEURONS; i++) {
                    //System.out.println(sample + " : " + i);
                    target[i] = trainOutput[sample][i];
                }
                feedForward();
                backPropagate();
            }
        } // epoch

        System.out.println("\nfinished testing "+ epoches + " epochs in "+((System.currentTimeMillis() - time)
                /1000)+" seconds");
        System.out.println("\nafter training");
        System.out.println("Network test error is " + getTrainingStats(test) + "% correct.");

        //System.out.println("\nTest network against original input:");
        //testNetworkTraining(samples);

        //System.out.println("\nTest network against noisy input:");
        //testNetworkWithNoise1();

        return;
    }

    private double getTrainingStats(ArrayList<Sample> samples)
    {
        double sum = 0.0;
        for(Sample s : samples)
        {
//            for(int j = 0; j < INPUT_NEURONS; j++)
//            {
//                inputs[j] = trainInputs[i][j];
//            } // j
            inputs[0] = s.X1;
            inputs[1] = s.X2;
            int sample = s.expectedClass;

            for(int j = 0; j < OUTPUT_NEURONS; j++)
            {
                target[j] = trainOutput[sample][j];
            } // j

            feedForward();

            if(maximum(actual) == maximum(target)){
                sum += 1;
            }else{
//                System.out.println(inputs[0] + "\t" + inputs[1]);
//                System.out.println("actual: " + maximum(actual) + "\ttarget: " + maximum(target));
//                System.out.println();
            }
        } // i

        return ((double)sum / (double)MAX_SAMPLES * 100.0);
    }

    private void testNetworkTraining(ArrayList<Sample> samples)
    {
        // This function simply tests the training vectors against network.
        for(Sample s : samples)
        {
//            for(int j = 0; j < INPUT_NEURONS; j++)
//            {
//                inputs[j] = trainInputs[i][j];
//            } // j
            inputs[0] = s.X1;
            inputs[1] = s.X2;

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(inputs[j] + "\t");
            } // j

            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    /**
     * as decribed in class, take the max output
     * @param outputVector
     * @return
     */
    private int maximum(double[] outputVector)
    {
        // This function returns the maxIndex of the maximum of outputVector().
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
    private void feedForward()
    {
        double sum = 0.0;

        // Calculate input to hidden layer.
        for(int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputs[inp] * inputToHiddenWeights[inp][hid];
            } // inp

            sum += bias * inputToHiddenWeights[INPUT_NEURONS][hid]; // Add in bias.
            hidden[hid] = sigmoid(sum);
        } // hid
        if(numebrOfHiddenLayers>1){
            for (int i = 0; i < numebrOfHiddenLayers; i++) {
                for (int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++) {
                    sum = 0.0;
                    for (int inp = 0; inp < numberOfHiddenNeuronsPerLayer; inp++) {
                        sum += hidden[inp] * hiddenLayers[i][inp][hid];
                    } // inp
                    sum += bias * hiddenLayers[i][numberOfHiddenNeuronsPerLayer][hid]; // Add in bias.
                    hidden[hid] = sigmoid(sum);
                } // hid
            }
        }
        // Calculate the hidden to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++)
            {
                sum += hidden[hid] * hiddenToOutputWeights[hid][out];
            } // hid

            sum += bias * hiddenToOutputWeights[numberOfHiddenNeuronsPerLayer][out]; // Add in bias.
            actual[out] = sigmoid(sum);
        } // out
        return;
    }

    private void backPropagate()
    {
        // Calculate the output layer error (step 3 for output cell).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out]);
        }

        // Calculate the hidden layer error (step 3 for hidden cell).
        for(int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++)
        {
            errh[hid] = 0.0;
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                errh[hid] += erro[out] * hiddenToOutputWeights[hid][out];
            }
            errh[hid] *= sigmoidDerivative(hidden[hid]);
        }
        // Update the weights for the output layer (step 4).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++)
            {
                hiddenToOutputWeights[hid][out] += (lRate * erro[out] * hidden[hid]);
            } // hid
            hiddenToOutputWeights[numberOfHiddenNeuronsPerLayer][out] += (lRate * erro[out] * bias); // Update the bias.
        } // out

        // Update the weights for the hidden layer (step 4).
        for(int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                inputToHiddenWeights[inp][hid] += (lRate * errh[hid] * inputs[inp]);
            } // inp
            inputToHiddenWeights[INPUT_NEURONS][hid] += (lRate * errh[hid] * bias); // Update the bias.
        } // hid
        return;
    }

    private void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < numberOfHiddenNeuronsPerLayer; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                inputToHiddenWeights[inp][hid] = new Random().nextDouble()*2 - 1;
                //inputToHiddenWeights[inp][hid] = 1;

            } // hid
        } // inp

        for(int hid = 0; hid <= numberOfHiddenNeuronsPerLayer; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                hiddenToOutputWeights[hid][out] = new Random().nextDouble()*2 - 1;
                //hiddenToOutputWeights[hid][out] = 1;
            } // out
        } // hid
        if(numebrOfHiddenLayers>1){
            for (int i = 0; i < numebrOfHiddenLayers-1; i++)
            {
                for (int hid = 0; hid <= numberOfHiddenNeuronsPerLayer; hid++) // Do not subtract 1 here.
                {
                    for (int hid2 = 0; hid2 < numberOfHiddenNeuronsPerLayer; hid2++) {
                        // Assign a random weight value between -0.5 and 0.5
                        hiddenLayers[i][hid][hid2] = new Random().nextDouble() * 2 - 1;
                        //hiddenToOutputWeights[hid][out] = 1;
                    } // out
                } // hid
            }
        }
        return;
    }

    private double sigmoid(double outputSum)
    {
        return (1.0 / (1.0 + Math.exp(-outputSum)));
    }

    private double sigmoidDerivative(double val)
    {
        return (val * (1.0 - val));
    }

    public void run(){NeuralNetwork();}

    public static void main(String[] args)
    {
        new BackPropNetwork().run();
    }

}