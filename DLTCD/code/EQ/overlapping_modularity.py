from collections import defaultdict

"""
    paper:<<Detect overlapping and hierarchical community structure in networks>>
    G:a graph of network (create by netwokx)
    cover:community [[com1],[com2],[com3]]
"""

def cal_EQ(cover,G):

    vertex_neighbors = defaultdict(lambda:list())
    nodes = G.nodes()
    for v in nodes:
        vertex_neighbors[v] = list(G.neighbors(v))

    vertex_community = defaultdict(lambda:set())
    for i,c in enumerate(cover):
        for v in c:
            vertex_community[v].add(i)
            
    m = G.number_of_edges()

    total = 0.0
    for c in cover:
        for i in c:
            o_i = len(vertex_community[i])
            k_i = len(vertex_neighbors[i])
            for j in c:
                o_j = len(vertex_community[j])
                k_j = len(vertex_neighbors[j])
                if i > j:
                    continue
                t = 0.0
                if j in vertex_neighbors[i]:
                    t += 1.0/(o_i*o_j)
                t -= k_i*k_j/(2*m*o_i*o_j)
                if i == j:
                    total += t
                else:
                    total += 2*t
    
    return round(total/(2*m),4)