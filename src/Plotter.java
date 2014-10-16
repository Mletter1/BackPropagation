import org.math.plot.Plot2DPanel;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

/**
 * Created by matthewletter on 10/15/14.
 */
public class Plotter {
//    private static void generalize(ArrayList<Sample> cls1,ArrayList<Sample> cls2,ArrayList<Sample> allLearningClasses,
//                                   ArrayList<Sample> allTestingClasses) {
//        // create your PlotPanel (you can use it as a JPanel)
//        Plot2DPanel plot = new Plot2DPanel();
//        // define the legend position
//        plot.addLegend("SOUTH");
//        double learningRate = .25;
//        for (int j = 0; j < 50; j++) {
//            BackPropNetwork p = new BackPropNetwork();
//            p.w0 = p.rnd.nextDouble();
//            p.w1 = p.rnd.nextDouble();
//            p.w2 = p.rnd.nextDouble();
//            p.maxIterations = 1;
//            p.learningRate = learningRate;
//            System.out.println("starting| w0:" + p.w0 + " w1:" + p.w1 + " w2:" + p.w2);
//            int length = 100;
//            double[] x1 = new double[length];
//            double[] x2 = new double[length];
//            double[] y = new double[length];
//
//
//            for (int i = 0; i < length; i++) {
//                x1[i] = p.learn(allLearningClasses);
//                x2[i] = p.test(allTestingClasses);
//                y[i] = i;
//            }
//            plot.addLinePlot("learning", Color.BLUE, y, x1);
//            plot.addLinePlot("testing", Color.RED, y, x2);
//        }
//
//        // put the PlotPanel in a JFrame like a JPanel
//        JFrame frame = new JFrame("class1 vs class2");
//        frame.setSize(1000, 1000);
//        frame.setContentPane(plot);
//        frame.setVisible(true);
//        frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);
//    }
}
