import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class GreyKmeans {
	public static void main(String[] args) {
		if (args.length < 3) {
			System.out
					.println("[Usage]java GreyKmeans <graph filename> <k> <max iteration>");
			return;
		}
		long startTime = System.currentTimeMillis();
		System.out.println("reading data...");
		Map<Integer, int[]> verAdjList = readData(args[0]);
		System.out.println("transforming to edge adjacent list...");
		Map<Edge, Set<Edge>> edgeAdjList = transformToEdgeAdjList(verAdjList);
		System.out.println("running clustering algorithm...");
		int k = Integer.parseInt(args[1]);
		int maxIter = Integer.parseInt(args[2]);
		// Collection<Set<Edge>> edgeClusters = kmedoids(edgeAdjList, k,
		// maxIter);
//		Collection<Set<Edge>> edgeClusters = kmeans(edgeAdjList, k, maxIter);
		Collection<Set<Edge>> edgeClusters = kmeansByWeka(edgeAdjList, k, maxIter);
		System.out.println("transforming edge clusters to vertex clusters...");
		Set<Set<Integer>> vertexClusters = transformToVertexClusters(edgeClusters);
		long endTime = System.currentTimeMillis();
		System.out.printf("Time cost: %.1f second(s).\n",
				((endTime - startTime) / 1000.0));
		System.out.println("found " + vertexClusters.size() + " clusters:");
		printClusters(vertexClusters);
	}

	private static Map<Integer, int[]> readData(String filename) {
		Map<Integer, int[]> resultMap = new HashMap<Integer, int[]>();
		BufferedReader reader = null;
		int nVer = 0;
		int nEdge = 0;
		try {
			reader = new BufferedReader(new FileReader(filename));
			String line = null;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.trim().split(" ");
				int key = Integer.parseInt(tokens[0]);
				int[] oneList = new int[tokens.length - 1];
				for (int i = 1; i < tokens.length; i++)
					oneList[i - 1] = Integer.parseInt(tokens[i]);
				resultMap.put(key, oneList);
				nVer++;
				nEdge += oneList.length;
			}
			reader.close();
			reader = null;
			nEdge /= 2;
		} catch (IOException e) {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e1) {}
			}
		}
		System.out.println("vertices: " + nVer + " edges: " + nEdge);
		return resultMap;
	}

	private static Map<Edge, Set<Edge>> transformToEdgeAdjList(
			Map<Integer, int[]> verAdjList) {
		Map<Edge, Set<Edge>> edgeAdjList = new HashMap<Edge, Set<Edge>>();
		for (Integer key : verAdjList.keySet()) {
			int[] verNeighbors = verAdjList.get(key);
			List<Edge> tempEdgeSet = new ArrayList<Edge>();
			for (int verNb : verNeighbors) {
				Edge edge = new Edge(key, verNb);
				if (!edgeAdjList.containsKey(edge)) {
					edgeAdjList.put(edge, new HashSet<Edge>());
				}
				tempEdgeSet.add(edge);
			}
			for (Edge edge1 : tempEdgeSet) {
				for (Edge edge2 : tempEdgeSet) {
					if (!edge2.equals(edge1))
						edgeAdjList.get(edge1).add(edge2);
				}
			}
		}
		int nEdge = edgeAdjList.size();
		int nLink = 0;
		for (Edge key : edgeAdjList.keySet())
			nLink += edgeAdjList.get(key).size();
		System.out.println("edges: " + nEdge + " edge links: " + (nLink / 2));
		return edgeAdjList;
	}

	private static Collection<Set<Edge>> kmedoids(
			final Map<Edge, Set<Edge>> edgeAdjList, int k, int maxIter) {
		Map<Edge, Set<Edge>> clusters = new HashMap<Edge, Set<Edge>>();
		Set<Edge> centers = new HashSet<Edge>();
		int i = 0;
		Set<Edge> edgeSet = edgeAdjList.keySet();
		System.out.println("deciding initial centers...");
		while (i < k) {
			for (Edge edge : edgeSet) {
				if (Math.random() > 0.5 && !centers.contains(edge)) {
					centers.add(edge);
					i++;
					if (i == k) break;
				}
			}
		}
		for (i = 0; i < maxIter; i++) {
			System.out.println("iteration " + (i + 1));
			System.out.println("building initial clusters...");
			clusters.clear();
			for (Edge center : centers) {
				Set<Edge> cluster = new HashSet<Edge>();
				cluster.add(center);
				clusters.put(center, cluster);
			}
			System.out.println("finding closest center for each edge...");
			for (Edge edge : edgeSet) {
				double minDist = Double.POSITIVE_INFINITY;
				Edge closestCenter = null;
				for (Edge center : centers) {
					double d = distance(edge, center, edgeAdjList);
					if (d < minDist) {
						minDist = d;
						closestCenter = center;
					}
				}
				clusters.get(closestCenter).add(edge);
			}
			System.out.println("deciding new center for each cluster...");
			final Set<Edge> newCenters = new HashSet<Edge>();
			for (Set<Edge> cluster : clusters.values()) {
				double minDist = Double.POSITIVE_INFINITY;
				Edge newCenter = null;
				for (Edge testEdge : cluster) {
					double dist = 0.0;
					for (Edge edge : cluster) {
						if (edge.equals(testEdge)) continue;
						double d = distance(edge, testEdge, edgeAdjList);
						dist += d;
					}
					if (dist < minDist) {
						minDist = dist;
						newCenter = testEdge;
					}
				}
				newCenters.add(newCenter);
			}
			System.out.println("old center: " + centers);
			System.out.println("new center: " + newCenters);
			centers.removeAll(newCenters);
			if (centers.isEmpty()) {
				break;
			}
			centers = newCenters;
		}
		System.out.println("done after " + (i == maxIter ? maxIter : i + 1)
				+ " iteration(s).");
		return clusters.values();
	}

	private static double distance(Edge e1, Edge e2,
			Map<Edge, Set<Edge>> edgeAdjList) {
		Set<Edge> set1 = edgeAdjList.get(e1);
		Set<Edge> set2 = edgeAdjList.get(e2);
		HashSet<Edge> unionSet = new HashSet<Edge>(set1);
		int n = unionSet.size();
		unionSet.addAll(set2);
		double[] v1 = new double[n];
		double[] v2 = new double[n];
		Iterator<Edge> iter = unionSet.iterator();
		for (int i = 0; i < n; i++) {
			Edge edge = iter.next();
			if (set1.contains(edge)) {
				v1[i] = 1;
			}
			if (set2.contains(edge)) {
				v2[i] = 1;
			}
		}
		double[] s = GreyRelationalAnalysis.analyze(new double[][] { v1, v2 },
				0.5, GreyRelationalAnalysis.NORM_OPERATOR_TYPE.INTERVAL, true);
		double dist = 1.0 - s[1];
		return dist;
	}

	private static Collection<Set<Edge>> kmeans(
			final Map<Edge, Set<Edge>> edgeAdjList, int k, int maxIter) {
		System.out.println("initialing edge vectors...");
		List<Edge> edgeList = new ArrayList<Edge>(edgeAdjList.keySet());
		Collections.sort(edgeList);
		int nEdge = edgeList.size();
		Map<Edge, Integer> edge2IdMap = new HashMap<Edge, Integer>();
		int i = 0;
		for (i = 0; i < nEdge; i++) {
			Edge edge = edgeList.get(i);
			edge2IdMap.put(edge, i);
		}

		double[][] data = new double[nEdge][nEdge];
		for (i = 0; i < nEdge; i++) {
			Edge edge = edgeList.get(i);
			Set<Edge> neighbors = edgeAdjList.get(edge);
			for (Edge nb : neighbors) {
				int id = edge2IdMap.get(nb);
				data[i][id] = 1.0;
			}
		}

		System.out.println("deciding initial centers...");
		double[][] centers = new double[k][];
		Set<Integer> selectedIndices = new HashSet<Integer>();
		Random random = new Random();
		for (i = 0; i < k; i++) {
			int j = random.nextInt(nEdge);
			while (selectedIndices.contains(j))
				j = random.nextInt(nEdge);
			centers[i] = Arrays.copyOf(data[j], nEdge);
			selectedIndices.add(j);
		}

		Map<Integer, Set<Integer>> clusters = new HashMap<Integer, Set<Integer>>();
		for (i = 0; i < maxIter; i++) {
			System.out.println("iteration " + (i + 1));
			System.out.println("finding closest center for each edge...");
			for (int j = 0; j < nEdge; j++) {
				double[] ev = data[j];
				double minDist = Double.POSITIVE_INFINITY;
				int closestCenter = -1;
				for (int p = 0; p < k; p++) {
					double d = distance(ev, centers[p]);
					if (d < minDist) {
						minDist = d;
						closestCenter = p;
					}
				}
				Set<Integer> cluster = clusters.get(closestCenter);
				if (cluster == null) {
					cluster = new HashSet<Integer>();
					clusters.put(closestCenter, cluster);
				}
				cluster.add(j);
			}

			System.out.println("deciding new center for each cluster...");
			double[][] newCenters = new double[k][];
			int q = 0;
			for (Set<Integer> cluster : clusters.values()) {
				double[] means = new double[nEdge];
				for (int j = 0; j < nEdge; j++) {
					for (Integer p : cluster) {
						means[j] += data[p][j];
					}
					means[j] /= nEdge;
				}
				newCenters[q] = means;
				q++;
			}
			double dist = 0.0;
			for (int j = 0; j < k; j++) {
				dist += distance(newCenters[j], centers[j]);
			}
			if (dist < nEdge * 1E-3) {
				break;
			}
			centers = newCenters;
		}
		for (i = 0; i < centers.length; i++) {
			double[] center = centers[i];
			System.out.println("center " + i + ": " + Arrays.toString(center));
		}
		System.out.println("done after " + (i == maxIter ? maxIter : i + 1)
				+ " iteration(s).");
		Collection<Set<Integer>> clusterSets = clusters.values();
		Set<Set<Edge>> retClusters = new HashSet<Set<Edge>>();
		for (Set<Integer> cluster : clusterSets) {
			Set<Edge> retCluster = new HashSet<Edge>();
			for (Integer j : cluster) {
				Edge edge = edgeList.get(j);
				retCluster.add(edge);
			}
			retClusters.add(retCluster);
		}
		return retClusters;
	}

	private static double distance(double[] v1, double[] v2) {
		double[] s = GreyRelationalAnalysis.analyze(new double[][] { v1, v2 },
				0.5, GreyRelationalAnalysis.NORM_OPERATOR_TYPE.INTERVAL, true);
		double dist = 1.0 - s[1];
		return dist;
	}

	private static Set<Set<Integer>> transformToVertexClusters(
			Collection<Set<Edge>> edgeClusters) {
		Set<Set<Integer>> vertexClusters = new HashSet<Set<Integer>>();
		Set<Integer> overlappingVertices = new HashSet<Integer>();
		for (Set<Edge> cluster : edgeClusters) {
			Set<Integer> vertexCluster = new HashSet<Integer>();
			for (Edge edge : cluster) {
				vertexCluster.add(edge.i);
				vertexCluster.add(edge.j);
				for (Set<Integer> vc : vertexClusters) {
					if (vc.contains(edge.i)) {
						overlappingVertices.add(edge.i);
					}
					if (vc.contains(edge.j)) {
						overlappingVertices.add(edge.j);
					}
				}
			}
			vertexClusters.add(vertexCluster);
		}
		System.out.println("overlapping vertices: " + overlappingVertices);
		return vertexClusters;
	}

	private static Collection<Set<Edge>> kmeansByWeka(
			final Map<Edge, Set<Edge>> edgeAdjList, int k, int maxIter) {
		System.out.println("initialing edge vectors...");
		List<Edge> edgeList = new ArrayList<Edge>(edgeAdjList.keySet());
		Collections.sort(edgeList);
		int nEdge = edgeList.size();
		Map<Edge, Integer> edge2IdMap = new HashMap<Edge, Integer>();
		int i = 0;
		for (i = 0; i < nEdge; i++) {
			Edge edge = edgeList.get(i);
			edge2IdMap.put(edge, i);
		}
		double[][] data = new double[nEdge][nEdge];
		for (i = 0; i < nEdge; i++) {
			Edge edge = edgeList.get(i);
			Set<Edge> neighbors = edgeAdjList.get(edge);
			for (Edge nb : neighbors) {
				int id = edge2IdMap.get(nb);
				double jaccardDist = calcJaccardDistance(neighbors, edgeAdjList.get(edgeList.get(id)));
				data[i][id] = jaccardDist;
			}
		}

		System.out.println("transforming data into Weka format...");
		FastVector attributes = new FastVector();
		for (i = 0; i < nEdge; i++) {
			Attribute attr = new Attribute("attr" + i);
			attributes.addElement(attr);
		}
		Instances dataset = new Instances("dataset", attributes, data.length);
		for (i = 0; i < nEdge; i++) {
			Instance instance = new SparseInstance(1.0, data[i]);
			dataset.add(instance);
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
			Map<Integer, Set<Edge>> clusters = new HashMap<Integer, Set<Edge>>();
			for (i = 0; i < nEdge; i++) {
				Edge edge = edgeList.get(i);
				int clusterId = assignments[i];
				Set<Edge> cluster = clusters.get(clusterId);
				if (cluster == null) {
					cluster = new HashSet<Edge>();
					clusters.put(clusterId, cluster);
				}
				cluster.add(edge);
			}
			return clusters.values();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	private static double calcJaccardDistance(Set<Edge> x, Set<Edge> y) {
		Set<Edge> union = new HashSet<Edge>(x);
		Set<Edge> intersect = new HashSet<Edge>(x);
		union.addAll(y);
		intersect.retainAll(y);
		double dist = intersect.size() * 1.0 / union.size();
		return dist;
	}

	private static void printClusters(Collection<Set<Edge>> clusters) {
		int i = 1;
		for (Set<Edge> cluster : clusters) {
			System.out.println("cluster " + i++ + ": " + cluster);
		}
	}

	private static void printClusters(Set<Set<Integer>> clusters) {
		int i = 1;
		for (Set<Integer> cluster : clusters) {
			System.out.println("cluster " + i++ + ": " + cluster);
		}
	}
}