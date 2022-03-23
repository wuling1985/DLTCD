import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.NormalizableDistance;

public class GreyDistanceFunction extends EuclideanDistance {
	@Override
	public double distance(Instance first, Instance second) {
		double[] s = GreyRelationalAnalysis
				.analyze(
						new double[][] { first.toDoubleArray(),
								second.toDoubleArray() }, 0.5,
						GreyRelationalAnalysis.NORM_OPERATOR_TYPE.INTERVAL,
						true);
		double dist = 1.0 - s[1];
		return dist;
	}
}
