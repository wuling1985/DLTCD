
class Edge implements Comparable{
	int i;
	int j;
	public Edge(int i, int j){
		if (i <= j){
			this.i = i;
			this.j = j;
		}else{
			this.i = j;
			this.j = i;
		}
	}
	
	public String toString(){
		return "<" + i + "," + j + ">";
	}
	
	public boolean equals(Object other){
		if (other == null) return false;
		Edge otherEdge = (Edge)other;
		return i == otherEdge.i && j == otherEdge.j;
	}
	
	public int hashCode(){
		return i << 16 + j;
	}
	
	public int compareTo(Object other){
		if (other == null) return 1;
		Edge otherEdge = (Edge)other;
		if (i < otherEdge.i) return -1;
		else if (i > otherEdge.i) return 1;
		else
			if (j < otherEdge.j) return -1;
			else if (j > otherEdge.j) return 1;
			else return 0;
	}
}