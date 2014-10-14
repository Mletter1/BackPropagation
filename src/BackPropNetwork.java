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

public class BackPropNetwork
{

    private final int INPUT_NEURONS = 2;
    private final int HIDDEN_NEURONS = 2;
    private final int OUTPUT_NEURONS = 4;

    private final double LEARN_RATE = 0.02;    // Rho.
    private final double NOISE_FACTOR = 0.2;
    private final int TRAINING_REPS = 100;

    // Input to Hidden Weights (with Biases).
    private double wih[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS];

    // Hidden to Output Weights (with Biases).
    private double who[][] = new double[HIDDEN_NEURONS + 1][OUTPUT_NEURONS];

    // Activations.
    private double inputs[] = new double[INPUT_NEURONS];
    private double hidden[] = new double[HIDDEN_NEURONS];
    private double target[] = new double[OUTPUT_NEURONS];
    private double actual[] = new double[OUTPUT_NEURONS];

    // Unit errors.
    private double erro[] = new double[OUTPUT_NEURONS];
    private double errh[] = new double[HIDDEN_NEURONS];

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
        for(int epoch = 0; epoch < TRAINING_REPS; epoch++)
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

        System.out.println("\nfinished testing "+TRAINING_REPS+ " epochs in "+((System.currentTimeMillis() - time)
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

    private void testNetworkWithNoise1()
    {
        // This function adds a random fractional value to all the training
        // inputs greater than zero.
        DecimalFormat dfm = new java.text.DecimalFormat("###0.0");

        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j] + (new Random().nextDouble() * NOISE_FACTOR);
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(dfm.format(((inputs[j] * 1000.0) / 1000.0)) + "\t");
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
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputs[inp] * wih[inp][hid];
            } // inp

            sum += wih[INPUT_NEURONS][hid]; // Add in bias.
            hidden[hid] = sigmoid(sum);
        } // hid

        // Calculate the hidden to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                sum += hidden[hid] * who[hid][out];
            } // hid

            sum += who[HIDDEN_NEURONS][out]; // Add in bias.
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
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            errh[hid] = 0.0;
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                errh[hid] += erro[out] * who[hid][out];
            }
            errh[hid] *= sigmoidDerivative(hidden[hid]);
        }

        // Update the weights for the output layer (step 4).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                who[hid][out] += (LEARN_RATE * erro[out] * hidden[hid]);
            } // hid
            who[HIDDEN_NEURONS][out] += (LEARN_RATE * erro[out]); // Update the bias.
        } // out

        // Update the weights for the hidden layer (step 4).
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                wih[inp][hid] += (LEARN_RATE * errh[hid] * inputs[inp]);
            } // inp
            wih[INPUT_NEURONS][hid] += (LEARN_RATE * errh[hid]); // Update the bias.
        } // hid
        return;
    }

    private void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                wih[inp][hid] = new Random().nextDouble()*2 - 1;
                //wih[inp][hid] = 1;

            } // hid
        } // inp

        for(int hid = 0; hid <= HIDDEN_NEURONS; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                who[hid][out] = new Random().nextDouble()*2 - 1;
                //who[hid][out] = 1;
            } // out
        } // hid
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