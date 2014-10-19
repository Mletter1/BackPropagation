/**
 * Created by matthewletter on 10/7/14.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.text.DecimalFormat;
import java.util.Scanner;

public class Network
{
    //Example_4x6x16 http://mnemstudio.org/neural-networks-4x6x14.htm
    private static final int INPUT_NEURONS = 2;
    private static final int HIDDEN_NEURONS = 2;
    private static final int OUTPUT_NEURONS = 1;

    private static final double LEARN_RATE = .05;    // Rho.
    private static final double NOISE_FACTOR = 0.45;
    private static final int TRAINING_REPS = 10;

    // Input to Hidden Weights (with Biases).
    private static double wih[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS];

    // Hidden to Output Weights (with Biases).
    private static double who[][] = new double[HIDDEN_NEURONS + 1][OUTPUT_NEURONS];

    // Activations.
    private static double inputs[] = new double[INPUT_NEURONS];
    private static double hidden[] = new double[HIDDEN_NEURONS];
    private static double target = 0;
    private static double actual[] = new double[OUTPUT_NEURONS];

    // Unit errors.
    private static double erro[] = new double[OUTPUT_NEURONS];
    private static double errh[] = new double[HIDDEN_NEURONS];

    private static final int MAX_SAMPLES = 200;

    /**
     * used to parse the provided text files
     * @param f file
     * @return ArrayList of Sample
     */
    public static ArrayList<Sample> parseFile(File f){
        Scanner scanner;
        String[] sA;
        String s;
        ArrayList<Sample> samples = new ArrayList<Sample>();
        try {
            scanner = new Scanner(f);
            s = scanner.nextLine();
            s = s.replaceAll("\\s+"," ");
            System.out.println(s);

            while(scanner.hasNext()){
                s = s.replaceAll("\\s+"," ");
                sA = s.split(" ");

                if(sA.length==4) {
                    samples.add(new Sample(Integer.parseInt(sA[0]),
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

    private static void NeuralNetwork()
    {
        //class 1
        File f1 = new File("/Users/matthewletter/Documents/BackPropagation/data/TrainingData.txt");
        ArrayList<Sample> train = parseFile(f1);

        System.out.println(""+train.size()+"");
        int sample = 0;

        assignRandomWeights();

        // Train the network.
        for(int epoch = 0; epoch < TRAINING_REPS; epoch++)
        {
            int count = 0;
            for(Sample s : train) {
                //if(count == 3)break;
                count++;
                inputs[0] = s.X1;
                inputs[1] = s.X2;
                target = s.expectedClass;
                feedForward();
                backPropagate();
            }

        } // epoch

        getTrainingStats(train);

//        System.out.println("\nTest network against original input:");
//        testNetworkTraining(train);
//
//        System.out.println("\nTest network against noisy input:");
        //testNetworkWithNoise1(train);

        return;
    }


    private static void feedForward()
    {
        double sum = 0.0;

        // Calculate input to nodeOutputs layer.
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                //System.out.println("weight hid ="+ wih[inp][hid]);
                sum += inputs[inp] * wih[inp][hid];
                //System.out.println("sum ="+ sum);
            } // inp

            //sum += wih[INPUT_NEURONS][hid]; // Add in bias.
            hidden[hid] = sum;
//            nodeOutputs[hid] = sigmoid(sum);

        } // hid

        // Calculate the nodeOutputs to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                //System.out.println("weight out ="+ wih[hid][out]+" times:"+nodeOutputs[hid]);
                sum += hidden[hid] * who[hid][out];
                //System.out.println("sum ="+ sum);
            } // hid

            sum += who[HIDDEN_NEURONS][out]; // Add in bias.
            actual[out] = sum;
//            actual[out] = sigmoid(sum);
            //System.out.println("calculated nodeOutputs to out ="+ actual[out]);
            System.out.println("calculated actual ="+sum);
        } // out
        return;
    }

    private static void backPropagate()
    {
        // Calculate the output layer error (step 3 for output cell).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            System.out.println("target ="+target);
            erro[out] = (target - actual[out]);

        }

        // Calculate the nodeOutputs layer error (step 3 for nodeOutputs cell).
        for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
        {
            errh[hid] = 0.0;
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                errh[hid] += erro[out] * who[hid][out];
                System.out.println("calculated in to errorH ="+errh[hid]);
            }
            //err[hid] *= sigmoidDerivative(nodeOutputs[hid]);
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

        // Update the weights for the nodeOutputs layer (step 4).
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

    private static void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                //wih[inp][hid] = new Random().nextDouble() - 0.5;
                wih[inp][hid] = 1;

            } // hid
        } // inp

        for(int hid = 0; hid <= HIDDEN_NEURONS; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                //who[hid][out] = new Random().nextDouble() - 0.5;
                who[hid][out] = 1;
            } // out
        } // hid
        return;
    }

    private static void getTrainingStats(ArrayList<Sample> train)
    {
        double sum = 0.0;
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(Sample s : train) {

                inputs[0] = s.X1;
                inputs[1] = s.X2;
                target = s.expectedClass;
//                for (int i = 0; i < OUTPUT_NEURONS; i++) {
//                    target[i] = trainOutput[sample][i];
//                } // i
                feedForward();

                if(maximum(actual) == target){
                    sum += 1;
                }else{
                    System.out.println(inputs[0] + "\t" + inputs[1]);
                    System.out.println("actual:"+maximum(actual) + "\t target:" + target);
                }
            }


        } // i

        System.out.println("Network is " + ((double)sum / (double)MAX_SAMPLES * 100.0) + "% correct.");

        return;
    }

    private static void testNetworkTraining(ArrayList<Sample> train)
    {
        // This function simply tests the training vectors against network.
        for(Sample s : train)
        {
//            for(int j = 0; j < INPUT_NEURONS; j++)
//            {
//                inputs[j] = trainInputs[i][j];
//            } // j
            inputs[0] = s.X1;
            inputs[1] = s.X2;


            feedForward();

            inputs[0] = s.X1;
            inputs[1] = s.X2;
            System.out.print(inputs[0] + "\t");
            System.out.print(inputs[1] + "\t");

            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static void testNetworkWithNoise1(ArrayList<Sample> train)
    {
        // This function adds a random fractional value to all the training
        // inputs greater than zero.
        DecimalFormat dfm = new java.text.DecimalFormat("###0.0");

        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(Sample s : train) {
                inputs[0] = s.X1+ (new Random().nextDouble() * NOISE_FACTOR);
                inputs[1] = s.X2+ (new Random().nextDouble() * NOISE_FACTOR);
            }

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(dfm.format(((inputs[j] * 1000.0) / 1000.0)) + "\t");
            } // j
            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static int maximum(final double[] vector)
    {
        // This function returns the index of the maximum of vector().
        int sel = 0;
        double max = vector[sel];

        for(int index = 0; index < OUTPUT_NEURONS; index++)
        {
            if(vector[index] > max){
                max = vector[index];
                sel = index;
            }
        }
        return sel;
    }
//    private static double sigmoid(final double val)
//    {
//        return (1.0 / (1.0 + Math.exp(-val)));
//    }
//
//    private static double sigmoidDerivative(final double val)
//    {
//        return (val * (1.0 - val));
//    }

    public static void main(String[] args)
    {
        NeuralNetwork();
        return;
    }

}
