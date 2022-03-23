import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class GreyKmeans2 {
	public static void main(String[] args) {
		if (args.length < 3) {
			System.out
					.println("[Usage]java GreyKmeans2 <vertex vector filename> <k> <max iteration> <output filename>");
			return;
		}
		long startTime = System.currentTimeMillis();
		System.out.println("reading data...");
		Map<Integer, double[]> verVectors = readData(args[0]);
		System.out.println("running clustering algorithm...");
		int k = Integer.parseInt(args[1]);
		int maxIter = Integer.parseInt(args[2]);
		Collection<Set<Integer>> vertexClusters = kmeansByWeka(verVectors, k, maxIter);
		long endTime = System.currentTimeMillis();
		System.out.printf("Time cost: %.1f second(s).\n",
				((endTime - startTime) / 1000.0));
		System.out.println("found " + vertexClusters.size() + " clusters:");
		printClusters(vertexClusters, args[3]);
	}

	private static Map<Integer, double[]> readData(String filename) {
		Map<Integer, double[]> resultMap = new HashMap<Integer, double[]>();
		BufferedReader reader = null;
		int nVer = 0;
		int nVector = 0;
		try {
			reader = new BufferedReader(new FileReader(filename));
			String line = null;
			line = reader.readLine();
			String[] tokens = line.trim().split("\\t");
			nVer = Integer.parseInt(tokens[0]);
			nVector = Integer.parseInt(tokens[1]);
			while ((line = reader.readLine()) != null) {
				tokens = line.trim().split("\\t");
				int key = Integer.parseInt(tokens[0]);
				double[] oneList = new double[tokens.length - 1];
				for (int i = 1; i < tokens.length; i++)
					oneList[i - 1] = Double.parseDouble(tokens[i]);
				resultMap.put(key, oneList);
			}
			reader.close();
			reader = null;
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally{
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e1) {}
			}
		}
		System.out.println("vertices: " + nVer + " vector length: " + nVector);
		return resultMap;
	}

	private static Collection<Set<Integer>> kmeansByWeka(
	final Map<Integer, double[]> verVectors, int k, int maxIter) {
		int nVertex = verVectors.size();
		double[][] data = new double[nVertex][nVertex];

		System.out.println("transforming data into Weka format...");
		FastVector attributes = new FastVector();
		for (Integer vId : verVectors.keySet()){
			Attribute attr = new Attribute("attr" + vId);
			attributes.addElement(attr);
		}
		Instances dataset = new Instances("dataset", attributes, nVertex);
		for (Integer vId : verVectors.keySet()){
			Instance instance = new SparseInstance(1.0, verVectors.get(vId));
			dataset.add(instance);
		}
		List<Integer> vertexList = new ArrayList<Integer>();
		for (Integer vId : verVectors.keySet()){
			vertexList.add(vId);
		}

		try {
			SimpleKMeans clusterer = new SimpleKMeans();
			String[] options = { "-N", String.valueOf(k), "-I", String.valueOf(maxIter), "-O" };
			System.out.println("begin clustering...");
			clusterer.setOptions(options);
			clusterer.setDistanceFunction(new GreyDistanceFunction());
			clusterer.buildClusterer(dataset);
			int[] assignments = clusterer.getAssignments();

			System.out.println("building predicted clusters...");
			Map<Integer, Set<Integer>> clusters = new HashMap<Integer, Set<Integer>>();
			for (int i = 0; i < nVertex; i++) {
				Integer vertex = vertexList.get(i);
				int clusterId = assignments[i];
				Set<Integer> cluster = clusters.get(clusterId);
				if (cluster == null) {
					cluster = new HashSet<Integer>();
					clusters.put(clusterId, cluster);
				}
				cluster.add(vertex);
			}
			return clusters.values();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	private static void printClusters(Collection<Set<Integer>> clusters, String outputfilename) {
		PrintWriter writer = null;
		
		try {
			writer = new PrintWriter(new BufferedWriter(new FileWriter(outputfilename)));
			int i = 1;
			for (Set<Integer> cluster : clusters) {
				System.out.println("cluster " + i++ + ": " + cluster);
				for (Integer vertex : cluster) {
					writer.print(" " + vertex);
				}
				writer.println();
				writer.flush();
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		finally {
			if (writer != null) {
				writer.close();
			}
		}
	}
}