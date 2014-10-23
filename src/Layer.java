import java.util.Random;

/**
 * Created by matthewletter on 10/16/14.
 */
public class Layer {
    //whats coming in
    public int INPUT_NEURONS = 2;
    //how many nodes do I have
    public int NUMBER_OF_NEURONS = 4;
    //my weights
    public double [][] inputToSelf = new double[INPUT_NEURONS + 1][NUMBER_OF_NEURONS];
    //my outputs
    public double nodeOutputs[] = new double[NUMBER_OF_NEURONS];
    //error
    public double err[] = new double[NUMBER_OF_NEURONS];

    /**
     * builds a layer of neurons
     * @param INPUT_NEURONS number of neurons coming into the layer
     * @param NUMBER_OF_NEURONS number of neurons in the layer
     */
    public Layer(int INPUT_NEURONS, int NUMBER_OF_NEURONS) {
        this.INPUT_NEURONS = INPUT_NEURONS;
        this.NUMBER_OF_NEURONS = NUMBER_OF_NEURONS;

        inputToSelf = new double[INPUT_NEURONS + 1][NUMBER_OF_NEURONS];
        nodeOutputs = new double[NUMBER_OF_NEURONS];
        err = new double[NUMBER_OF_NEURONS];

        assignRandomWeights();
    }

    /**
     * assign random weights to each neuron in the layer
     */
    private void assignRandomWeights() {
        //each intput gets 1 weight for every neuron
        for (int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for (int nodeNum = 0; nodeNum < NUMBER_OF_NEURONS; nodeNum++) {
                // Assign a random weight value between -0.5 and 0.5
                //inputToSelf[inp][nodeNum] = new Random().nextDouble() * 1 - .5;
                inputToSelf[inp][nodeNum] = 1;

            }
        }
    }

}
