/**
 * Created by matthewletter on 10/7/14.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

public class BackPropNetwork
{

    private final int INPUT_NEURONS = 2;
    public int numberOfHiddenLayers = 2;
    private final int OUTPUT_NEURONS = 4;

    public double lRate = 0.02;    // Rho.
    public int epoches = 100;


    // Input to Hidden Weights (with Biases).
    private double inputToHiddenWeights[][] = new double[INPUT_NEURONS + 1][numberOfHiddenLayers];

    // Hidden to Output Weights (with Biases).
    private double hiddenToOutputWeights[][] = new double[numberOfHiddenLayers + 1][OUTPUT_NEURONS];

    // Activations.
    private double inputs[] = new double[INPUT_NEURONS];
    private double hidden[] = new double[numberOfHiddenLayers];
    private double target[] = new double[OUTPUT_NEURONS];
    private double actual[] = new double[OUTPUT_NEURONS];

    // Unit errors.
    private double erro[] = new double[OUTPUT_NEURONS];
    private double errh[] = new double[numberOfHiddenLayers];

    private final int MAX_SAMPLES = 800;

    private int trainOutput[][] = new int[][]
                   {{1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0},
                    {0, 0, 0, 1}};

    private void NeuralNetwork()
    {
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
        for (int i = 0; i < numberOfHiddenLayers; i++) {

        }

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



    private void feedForward()
    {
        double sum = 0.0;

        // Calculate input to nodeOutputs layer.
        for(int hid = 0; hid < numberOfHiddenLayers; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputs[inp] * inputToHiddenWeights[inp][hid];
            } // inp

            sum += inputToHiddenWeights[INPUT_NEURONS][hid]; // Add in bias.
            hidden[hid] = sigmoid(sum);
        } // hid

        // Calculate the nodeOutputs to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < numberOfHiddenLayers; hid++)
            {
                sum += hidden[hid] * hiddenToOutputWeights[hid][out];
            } // hid

            sum += hiddenToOutputWeights[numberOfHiddenLayers][out]; // Add in bias.
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

        // Calculate the nodeOutputs layer error (step 3 for nodeOutputs cell).
        for(int hid = 0; hid < numberOfHiddenLayers; hid++)
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
            for(int hid = 0; hid < numberOfHiddenLayers; hid++)
            {
                hiddenToOutputWeights[hid][out] += (lRate * erro[out] * hidden[hid]);
            } // hid
            hiddenToOutputWeights[numberOfHiddenLayers][out] += (lRate * erro[out]); // Update the bias.
        } // out

        // Update the weights for the nodeOutputs layer (step 4).
        for(int hid = 0; hid < numberOfHiddenLayers; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                inputToHiddenWeights[inp][hid] += (lRate * errh[hid] * inputs[inp]);
            } // inp
            inputToHiddenWeights[INPUT_NEURONS][hid] += (lRate * errh[hid]); // Update the bias.
        } // hid
        return;
    }

    private void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < numberOfHiddenLayers; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                inputToHiddenWeights[inp][hid] = new Random().nextDouble()*2 - 1;
                //inputToHiddenWeights[inp][hid] = 1;

            } // hid
        } // inp

        for(int hid = 0; hid <= numberOfHiddenLayers; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                hiddenToOutputWeights[hid][out] = new Random().nextDouble()*2 - 1;
                //hiddenToOutputWeights[hid][out] = 1;
            } // out
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
    private double getTrainingStats(ArrayList<Sample> samples)
    {
        double sum = 0.0;
        for(Sample s : samples)
        {
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