# Dr Michaels
# Floyd_Warshall.py
# 11/6/25
# This file contains a Python implementation of a graph using adjacency matrix representation
# and the All Pairs Shortest Path algorithm Floyd Warshall, as well as sample graphs for testing
#
# In the matrix, graph[i][j] is the weight of the edge i -> j, float('inf') means "no edge",
# and graph[i][i] is 0 (the shortest path from a node to itself is the empty path).
# Negative edge weights are fine. A negative cycle is not: distances along it have no minimum.
# floyd_warshall does not stop on one, but has_negative_cycle() will tell you if one exists.
#
# floyd_warshall returns (dist, parent), like the single source algorithms in graph.py:
#   dist[i][j]   = length of the shortest path from i to j (float('inf') if there is no path)
#   parent[i][j] = the node just before j on that shortest path (None if i == j or there is no path)
# Use build_path(parent[i], i, j) to rebuild the route from i to j.
# For the adjacency LIST version (Dijkstra, Bellman-Ford, Johnson's), see graph.py


class Weighted_Graph:
    """TO-DO: add function doc here"""

    def __init__(self, num_nodes, counter=None):
        self.num_nodes = num_nodes
        self.graph = [ ]
        for i in range(num_nodes):
            if counter is not None:
                counter.check_timeout()
            self.graph.append([0 if i==x else float('inf') for x in range(num_nodes)])
        self.num_edges = 0

    def get_num_nodes(self):
        """TO-DO: add function doc here"""
        return self.num_nodes
    
    def get_edge(self, i, j):
        """TO-DO: add function doc here"""
        if i > -1 and i < self.num_nodes and j > -1 and j < self.num_nodes:
            return self.graph[i][j]
        else:
            return -1234567890 # a clear sign there is a problem
        
    def add_undirected_edge(self, i, j, c):
        """Adds the edges i -> j and j -> i, both with weight c (overwrites weights if they exist)
        Self loops (i == j) are ignored, they would overwrite the 0 on the diagonal
        NOTE: an undirected edge with a negative weight is a negative cycle (i -> j -> i)"""
        self.add_directed_edge(i, j, c)
        self.add_directed_edge(j, i, c)

    def add_directed_edge(self, i, j, c):
        """Adds the edge i -> j with weight c (overwrites the weight if the edge exists)
        Self loops (i == j) are ignored, they would overwrite the 0 on the diagonal"""
        if i != j and i > -1 and i < self.num_nodes and j > -1 and j < self.num_nodes:
            if self.graph[i][j] == float('inf'):
                self.num_edges += 1
            self.graph[i][j] = c

    def __str__(self):
        result = "Num Nodes: %d\nNum Edges: %d\n" % (self.num_nodes, self.num_edges)
        for i in range(self.num_nodes):
            result += "Node %d: " % i
            for j in range(self.num_nodes):
                if i != j and self.graph[i][j] < float('inf'):
                    result += "<%d, %d, %d> " % (i, j, self.graph[i][j])
            result += "\n"
        return result


def floyd_warshall(my_graph, verbose=True, counter=None):
    """Floyd Warshall: returns (dist, parent), two n x n matrices. my_graph is not modified.
      dist[i][j]   = length of the shortest path from i to j (float('inf') if there is no path)
      parent[i][j] = the node just before j on that shortest path (None if i == j or there is no path)
    After the iteration for k, dist[i][j] is the shortest path from i to j that only uses
    nodes 0..k as intermediate stops. Updating dist in place is safe: during iteration k,
    row k and column k never change (dist[k][k] is 0 unless there is a negative cycle).
    Whenever going through k is shorter, the path from i to j now ends the way the path from k to j
    does, so the node before j is parent[k][j]. That is the only change we need to remember routes.
    Runs in O(n^3). Set verbose=False to skip the matrix printed after every iteration.
    NOTE: if the graph has a negative cycle (has_negative_cycle(dist)), the parents are not meaningful"""

    num_nodes = my_graph.get_num_nodes()
    dist = [[my_graph.get_edge(i,j) for j in range(num_nodes)] for i in range(num_nodes)]
    # With no intermediate stops, the node before j on the path i -> j is i itself (if the edge exists)
    parent = [[i if (i != j and dist[i][j] < float('inf')) else None for j in range(num_nodes)] for i in range(num_nodes)]

    for k in range(num_nodes):
        if counter is not None:
            counter.check_timeout()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if counter is not None and j % 256 == 0:
                    counter.check_timeout()
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    if counter is not None:
                        counter.relaxations += 1
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        parent[i][j] = parent[k][j]
        if verbose:
            print("\n***********************\n")
            print("Finished with iteration where k = %d" % k)
            print_matrix(dist)
    return dist, parent

def has_negative_cycle(matrix):
    """Takes the distance matrix returned by floyd_warshall
    A node whose shortest path to itself is below 0 lies on a negative cycle"""
    return any(matrix[i][i] < 0 for i in range(len(matrix)))

def build_path(parent, source, target):
    """Rebuilds a path from one row of the parent matrix: build_path(parent[source], source, target)
    Starts at target and follows parents back until it reaches source
    Returns the path as a list [source, ..., target], or None if target cannot be reached from source
    (graph.py has a similar helper built into shortest_path)"""
    path = [target]
    node = target
    while node != source:
        node = parent[node]
        # None: no path. Longer than the graph: we are going around a negative cycle.
        if node is None or len(path) > len(parent):
            return None
        path.append(node)
    path.reverse()
    return path

def print_matrix(matrix):
    """TO-DO: add function doc here"""
    result = ""
    for row in matrix:
        for val in row:
            if val == float('inf'):
                result += " - "
            else:
                result += "%2d " % val
        result += "\n"
    print(result)

def print_parents(parent):
    """Prints the parent matrix. A dash means there is no parent"""
    result = ""
    for row in parent:
        for val in row:
            if val is None:
                result += " - "
            else:
                result += "%2d " % val
        result += "\n"
    print(result)

def make_test_one():
    test_one = Weighted_Graph(7)
    test_one.add_undirected_edge(0, 1, 2)
    test_one.add_undirected_edge(0, 2, 4)
    test_one.add_undirected_edge(0, 3, 1)
    test_one.add_undirected_edge(1, 3, 3)
    test_one.add_undirected_edge(1, 4, 10)
    test_one.add_undirected_edge(2, 3, 2)
    test_one.add_undirected_edge(2, 5, 5)
    test_one.add_undirected_edge(3, 4, 7)
    test_one.add_undirected_edge(3, 5, 8)
    test_one.add_undirected_edge(3, 6, 4)
    test_one.add_undirected_edge(4, 6, 6)
    test_one.add_undirected_edge(5, 6, 1)
    return test_one

def make_test_two():
    test_two = Weighted_Graph(9)
    test_two.add_directed_edge(0,1,5)
    test_two.add_directed_edge(0,3,7)
    test_two.add_directed_edge(1,2,2)
    test_two.add_directed_edge(1,4,-3)
    test_two.add_directed_edge(2,3,-2)
    test_two.add_directed_edge(2,5,3)
    test_two.add_directed_edge(3,1,1)
    test_two.add_directed_edge(3,4,1)
    test_two.add_directed_edge(4,5,2)
    test_two.add_directed_edge(4,7,3)
    test_two.add_directed_edge(5,6,2)
    test_two.add_directed_edge(5,8,6)
    test_two.add_directed_edge(6,3,-2)
    test_two.add_directed_edge(6,7,1)
    test_two.add_directed_edge(7,2,2)
    test_two.add_directed_edge(7,8,4)
    test_two.add_directed_edge(8,0,-1)
    test_two.add_directed_edge(8,4,1)
    return test_two

def make_test_three():
    test_three = Weighted_Graph(7)
    test_three.add_directed_edge(0, 1, 5)
    test_three.add_directed_edge(0, 3, 6)
    test_three.add_directed_edge(1, 2, 2)
    test_three.add_directed_edge(1, 3, 3)
    test_three.add_directed_edge(2, 0, 2)
    test_three.add_directed_edge(2, 6, 2)
    test_three.add_directed_edge(3, 0, 2)
    test_three.add_directed_edge(3, 4, 4)
    test_three.add_directed_edge(3, 5, 1)
    test_three.add_directed_edge(4, 6, 1)
    test_three.add_directed_edge(5, 0, 5)
    test_three.add_directed_edge(5, 2, 3)
    test_three.add_directed_edge(5, 6, 5)
    test_three.add_directed_edge(6, 0, 4)
    test_three.add_directed_edge(6, 2, 2)
    return test_three

def main():
    print("This file will test the floyd warshall algorithm and the matrix representation of a graph in Python")
    # test_one = make_test_one()
    # print(test_one)
    # print("\n****************************\n")
    # result_one = floyd_warshall(test_one)
    # print_matrix(result_one)

    test_graph = make_test_three()
    print(test_graph)
    dist, parent = floyd_warshall(test_graph)
    print("\n*********************\n")
    print("Distances:")
    print_matrix(dist)
    print("Parents (row i, column j: the node before j on the shortest path from i):")
    print_parents(parent)
    if has_negative_cycle(dist):
        print("This graph contains a negative cycle, the distances above are not meaningful")
    else:
        print("Shortest path from 4 to 1: %s (distance %s)" % (build_path(parent[4], 4, 1), dist[4][1]))

if __name__ == "__main__":
    main()
