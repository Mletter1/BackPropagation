/**
 * Created by matthewletter on 9/30/14.
 */
public class Sample {
    public double X1 = 0;
    public double X2 = 0;
    public int expectedClass = 0;
    public int index = 0;
    Sample(int expectedClass, int index, double X1, double X2 ){
        this.X1=X1;
        this.X2=X2;
        this.expectedClass=expectedClass;
        this.index=index;
    }
    Sample(double X1, double X2 ){
        this.X1=X1;
        this.X2=X2;
    }
}
