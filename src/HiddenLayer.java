import java.util.Random;

/**
 * Created by matthewletter on 10/16/14.
 */
public class HiddenLayer {

    public int INPUT_NEURONS = 2;
    public int NUMBER_OF_NEURONS = 1;
    public int OUTPUT_NEURONS = 4;

    //weights

    public HiddenLayer(int INPUT_NEURONS, int NUMBER_OF_NEURONS, int OUTPUT_NEURONS) {
        this.INPUT_NEURONS = INPUT_NEURONS;
        this.NUMBER_OF_NEURONS = NUMBER_OF_NEURONS;
        this.OUTPUT_NEURONS = OUTPUT_NEURONS;

        inputToHidden = new double[INPUT_NEURONS + 1][NUMBER_OF_NEURONS];
        hiddenToOutput = new double[NUMBER_OF_NEURONS + 1][OUTPUT_NEURONS];
        hidden = new double[NUMBER_OF_NEURONS];
        errh = new double[NUMBER_OF_NEURONS];

        assignRandomWeights();
    }

    //weights
    public double [][] inputToHidden = new double[INPUT_NEURONS + 1][NUMBER_OF_NEURONS];
    public double [][] hiddenToOutput = new double[NUMBER_OF_NEURONS + 1][OUTPUT_NEURONS];

    //outputs
    public double hidden[] = new double[NUMBER_OF_NEURONS];
    //err
    public double errh[] = new double[NUMBER_OF_NEURONS];

    private void assignRandomWeights() {
        for (int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for (int hid = 0; hid < NUMBER_OF_NEURONS; hid++) {
                // Assign a random weight value between -0.5 and 0.5
                inputToHidden[inp][hid] = new Random().nextDouble() * 2 - 1;
                //inputToHiddenWeights[inp][hid] = 1;

            } // hid
        } // inp
        for(int hid = 0; hid <= NUMBER_OF_NEURONS; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                hiddenToOutput[hid][out] = new Random().nextDouble()*2 - 1;
                //hiddenToOutputWeights[hid][out] = 1;
            } // out
        } // hid
    }

}
