import org.math.plot.Plot2DPanel;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

/**
 * Created by matthewletter on 10/15/14.
 */
public class Plotter {
    public static void generalize(double[] x1,double[] x2,double[] y) {
        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();
        // define the legend position
        plot.addLegend("SOUTH");
            plot.addLinePlot("learning", Color.BLUE, y, x1);
            plot.addLinePlot("testing", Color.RED, y, x2);
        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("class1 vs class2");
        frame.setSize(1000, 1000);
        frame.setContentPane(plot);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);
    }
    public void regions(double[] x1,double[] x2,double[] x3,double[] x4,double[] y1,double[] y2,
                                  double[] y3,double[] y4) {
        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();
        // define the legend position
        plot.addLegend("SOUTH");
        System.out.println(x1.length+" : "+x2.length);
        if(x1.length>0&&y1.length>0) {
            plot.addScatterPlot("class1", Color.RED, x1, y1);
        }
        if(x2.length>0&&y2.length>0) {
            plot.addScatterPlot("class2", Color.BLUE, x2, y2);
        }
        if(x3.length>0&&y3.length>0) {
            plot.addScatterPlot("class3", Color.GREEN, x3, y3);
        }
        if(x4.length>0&&y4.length>0) {
            plot.addScatterPlot("class4", Color.MAGENTA, x4, y4);
        }
        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("class1 vs class2");
        frame.setSize(1000, 1000);
        frame.setContentPane(plot);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);
    }
}
