import static java.lang.System.currentTimeMillis;
import static java.lang.System.out;

import java.util.Arrays;
import java.util.Random;

/**
 * 灰关联分析方法集合。
 * 
 * @author Kun Guo
 * @date 2015-2-20
 */
public class GreyRelationalAnalysis {
	public static final double EPSLON = 1e-6;

	public enum NORM_OPERATOR_TYPE {
		NORM_BY_FIRST, AVG, MAX, MIN, INTERVAL, ZERO_BY_FIRST
	}

	public static void main(String[] args) {
		double[][] data = { { 1, 2, 3, 4, 5 }, { 1, 3, 5, 7, 9 },
				{ 2, 4, 6, 8, 10 }, { 2, 4, 8, 16, 32 } };
		double[] grd;
		try {
			out.println("small static dataset test...");
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.NORM_BY_FIRST, true);
			out.println(Arrays.toString(grd));
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.AVG, true);
			out.println(Arrays.toString(grd));
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.MAX, true);
			out.println(Arrays.toString(grd));
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.MIN, true);
			out.println(Arrays.toString(grd));
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.INTERVAL, true);
			out.println(Arrays.toString(grd));
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.ZERO_BY_FIRST, true);
			out.println(Arrays.toString(grd));

			out.println("large random dataset test...");
			int n = 10000, m = 10000;
			out.println("n = " + n + ", m = " + m);
			data = new double[n][m];
			Random rand = new Random();
			for (int i = 0; i < n; i++)
				for (int j = 0; j < m; j++)
					data[i][j] = rand.nextDouble();
			long startTime = currentTimeMillis();
			grd = analyze(data, 0.5, NORM_OPERATOR_TYPE.NORM_BY_FIRST, true);
			long endTime = currentTimeMillis();
			out.printf("time cost: %.2f s\n", (endTime - startTime) / 1000.0);
			Runtime rt = Runtime.getRuntime();
			out.printf("total memory: %.1f MB, free memory: %.1f MB\n",
					rt.totalMemory() / (1024 * 1024.0), rt.freeMemory()
							/ (1024 * 1024.0));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * 规范化矩阵的列向量。直接在参数矩阵上操作，若要保留原有矩阵，需在调用前备份。
	 * 
	 * @param data
	 *            待规范化的矩阵
	 * @param operType
	 */
	public static void normalize(double[][] data, NORM_OPERATOR_TYPE operType) {
		int nRow = data.length;
		int nCol = data[0].length;
		switch (operType) {
		case NORM_BY_FIRST:
			for (int j = 0; j < nCol; j++) {
				double first = data[0][j];
				if (first <= EPSLON && first >= -EPSLON) first = EPSLON;
				for (int i = 0; i < nRow; i++) {
					data[i][j] /= first;
				}
			}
			break;
		case ZERO_BY_FIRST:
			for (int j = 0; j < nCol; j++) {
				double first = data[0][j];
				for (int i = 0; i < nRow; i++) {
					data[i][j] -= first;
				}
			}
			break;
		case AVG:
			for (int j = 0; j < nCol; j++) {
				double avg = 0.0;
				for (int i = 0; i < nRow; i++)
					avg += data[i][j];
				avg /= nRow;
				if (avg <= EPSLON && avg >= -EPSLON) avg = EPSLON;
				for (int i = 0; i < nRow; i++) {
					data[i][j] /= avg;
				}
			}
			break;
		case MAX:
			for (int j = 0; j < nCol; j++) {
				double max = Double.NEGATIVE_INFINITY;
				for (int i = 0; i < nRow; i++)
					if (data[i][j] > max) max = data[i][j];
				if (max <= EPSLON && max >= -EPSLON) max = EPSLON;
				for (int i = 0; i < nRow; i++) {
					data[i][j] /= max;
				}
			}
			break;
		case MIN:
			for (int j = 0; j < nCol; j++) {
				double min = Double.POSITIVE_INFINITY;
				for (int i = 0; i < nRow; i++)
					if (data[i][j] < min) min = data[i][j];
				if (min <= EPSLON && min >= -EPSLON) min = EPSLON;
				for (int i = 0; i < nRow; i++) {
					data[i][j] /= min;
				}
			}
			break;
		case INTERVAL:
			for (int j = 0; j < nCol; j++) {
				double max = Double.NEGATIVE_INFINITY;
				double min = Double.POSITIVE_INFINITY;
				for (int i = 0; i < nRow; i++) {
					if (data[i][j] > max) max = data[i][j];
					if (data[i][j] < min) min = data[i][j];
				}
				double interval = max - min;
				if (interval <= EPSLON && interval >= -EPSLON)
					interval = EPSLON;
				for (int i = 0; i < nRow; i++) {
					data[i][j] /= interval;
				}
			}
			break;
		}
	}

	/**
	 * 计算数据矩阵的每个行向量与比较序列（默认为矩阵的第1行）的灰关联系数。 直接在参数矩阵上操作，若要保留原有矩阵，需在调用前备份。
	 * 
	 * @param data
	 *            数据矩阵
	 * @param zita
	 *            分辨系数
	 * @param operType
	 *            量化算子
	 * @param diffEntropy
	 *            是否进一步求基于差异信息理论的灰关联系数
	 * @return 灰关联系数向量。注意：向量的第1个分量为参考序列与自身的灰关联系数，此值没有意义，不应使用。
	 * @throws Exception
	 */
	public static double[] analyze(final double[][] data, double zita,
			NORM_OPERATOR_TYPE operType, boolean diffEntropy) {
		int nRow = data.length;
		int nCol = data[0].length;
		// 规范化矩阵。
		normalize(data, operType);
		// 求两级最大差和两级最小差。
		double minminDelta = Double.POSITIVE_INFINITY;
		double maxmaxDelta = Double.NEGATIVE_INFINITY;
		for (int i = 1; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				double d = data[i][j] - data[0][j];
				if (d < 0.0) d = -d;
				if (d < minminDelta) minminDelta = d;
				if (d > maxmaxDelta) maxmaxDelta = d;
			}
		}
		// 计算灰关联系数。
		double[] gamma = new double[nRow];
		double numerator = minminDelta + zita * maxmaxDelta;
		if (numerator <= EPSLON && numerator >= -EPSLON) numerator = EPSLON;
		double[] r = new double[nCol];
		for (int i = 1; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				double d = data[i][j] - data[0][j];
				if (d < 0.0) d = -d;
				double denominator = d + zita * maxmaxDelta;
				if (denominator <= EPSLON && denominator >= -EPSLON)
					denominator = EPSLON;
				r[j] = numerator / denominator;//第一行为参考序列，第i行比较序列，两个序列在j点的灰关联系数
			}
			double sum = 0.0;
			for (int j = 0; j < nCol; j++)
				sum += r[j];
			gamma[i] = sum / nCol;//第一行为参考序列，第i行比较序列的灰关联度
			// 计算基于差异信息理论的灰关联系数。
			if (diffEntropy) {
				double iv = 0.0;
				for (int j = 0; j < nCol; j++) {
					double v = r[j] / sum;	//第i个比较序列的第j个点的灰色关联密度
					iv += (v * Math.log(v));//第i个比较序列的灰色关联熵
				}
				double E = -iv / Math.log(nCol);//第i个比较序列的熵关联度
				gamma[i] *= E;//灰关联度和熵关联度的乘积，第一行为参考序列，第i行比较序列的均衡相似度
			}
		}
		return gamma;
	}
}
