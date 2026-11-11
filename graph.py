# Dr Michaels
# This file contains an implementation of a graph class using adjacency lists
# As we are in python, for ease of searching we are using dictionaries for both node storage and edge storage
# This greatly speeds up our searching for any individual edge
# Both directed and undirected edges may be added into the graph
# Directed edges may form cycles (earlier versions of this file only allowed DAGs)
# Undirected edges (by definition) always create a cycle
# For either edge type, we will not allow multiple edges a -> b,
# We can change those metrics if we wish


import heapq
import random

class Graph:

    # Constructor for a weighted graph class (directed and/or undirected edges)
    # It takes no input
    # We will store our node names in a dictionary as keys
    # The value at each key will be a dictionary
    # The dictionary will hold node names as keys, with weights being the value at each key
    # This project only allows non-negative edge weights, so Dijkstra is safe for
    # every graph built through add_edge/add_undirected_edge.
    # NOTE: At most we can have n(n-1) directed edges in the graph, one per ordered pair (u, v)
    # (Back when this class only allowed DAGs the limit was n(n-1)/2, because any more edges
    # was guaranteed to give us at least one cycle)
    def __init__(self):
        self.nodes = {}
        self.num_nodes = 0
        self.num_edges = 0

    # Return the number of nodes in our graph
    def get_num_nodes(self):
        return self.num_nodes
    
    # Returns the number of edges in our graph
    def get_num_edges(self):
        return self.num_edges
    
    # Adds a node to our graph if it is not already in it
    def add_node(self, name):
        if not name in self.nodes:
            self.nodes[name] = {}
            self.num_nodes += 1

    # Checks to see if a given node is in our graph
    def has_node(self, name):
        if name in self.nodes:
            return True
        return False
    
    
    # We will return False if the edge node1 -> node2 already exists, and True otherwise
    # Cycles are allowed: node2 -> node1 may already exist (a 2-cycle), and node2 may already
    # reach node1 through other nodes. Earlier versions of this method rejected both of those
    # cases so that the graph stayed a DAG; that restriction was removed so we can build graphs
    # with strongly connected components.
    def check_edge(self, node1, node2):
        if node1 in self.nodes:
            if node2 in self.nodes[node1]:
                # print("Edge between %s and %s already exists" %(node1, node2))
                return False
        return True
    
    # Determines if node2 is reachable from node1 via a simple path
    # (add_edge no longer calls this, since cycles are allowed, but it is a handy reachability helper)
    def check_path(self, node1, node2):
        to_visit = [node1]
        has_visited = {}
        while len(to_visit) > 0:
            curr = to_visit.pop()
            has_visited[curr] = True
            for node in self.nodes[curr]:
                if node == node2:
                    return True
                if node not in has_visited:
                    to_visit.append(node)
        return False
    
    # Iterative DFS (note we proceed with the LAST adjacent node added, not the first, order is slightly different than a traditional DFS)
    # NOTE: node membership is checked directly (node in self.nodes) rather than a numeric
    # range check against num_nodes, and "visited" is tracked with a set rather than a list
    # indexed by node value. A list indexed by node value only works if node names happen to be
    # the exact contiguous integers 0..num_nodes-1 -- true for our current test files, but not
    # guaranteed in general (an isolated node that never appears in an edge, or non-contiguous
    # labels, would silently break it). Membership/set-based tracking has no such assumption.
    def get_component(self, node):
        if node not in self.nodes:
            return "Illegal node entered"
        nodes_to_travel = [node]
        visited_nodes = set()
        result = ""
        while len(nodes_to_travel) > 0:
            next_node = nodes_to_travel.pop()
            visited_nodes.add(next_node)
            result += "%s " % next_node
            for new_node in self.nodes[next_node]:
                if (new_node not in nodes_to_travel) and (new_node not in visited_nodes):
                    nodes_to_travel.append(new_node)
        return result

    # Recursive DFS
    # Will explore entire graph
    # NOTE: "found" is a set rather than a list indexed by node value, for the same reason as
    # get_component above -- no assumption that node names are contiguous integers from 0.
    def DFS(self):
        found = set()
        for node in self.nodes:
            if node not in found:
                self.DFS_visit(node, found)
                print("")

    def DFS_visit(self, node, found):
        found.add(node)
        print("%s" % node, end=" ")
        for neighbor in self.nodes[node]:
            if neighbor not in found:
                self.DFS_visit(neighbor, found)

    # Returns all nodes in DFS FINISH order: a node is appended to the list only after every
    # one of its descendants has already been explored, NOT when it is first discovered.
    # This is a different order than the one DFS() prints above. Look closely at where the
    # append happens -- why does its position matter?
    def dfs_finish_order(self):
        found = set()
        finish_order = []
        for node in self.nodes:
            if node not in found:
                self._dfs_finish_visit(node, found, finish_order)
        return finish_order

    def _dfs_finish_visit(self, node, found, finish_order):
        found.add(node)
        for neighbor in self.nodes[node]:
            if neighbor not in found:
                self._dfs_finish_visit(neighbor, found, finish_order)
        finish_order.append(node)

    # Returns a NEW Graph with every directed edge reversed (weights preserved).
    # Every node is copied over first, so a node with no edges at all still exists in the result
    def transpose(self):
        reversed_graph = Graph()
        for node in self.nodes:
            reversed_graph.add_node(node)
        for node in self.nodes:
            for neighbor, weight in self.nodes[node].items():
                reversed_graph.add_edge(neighbor, node, weight)
        return reversed_graph

    # Kosaraju's algorithm: returns a list of the strongly connected components (SCCs)
    # Each component is a list of nodes. A node on no cycle is a component by itself.
    #   Pass 1: DFS on the original graph, recording the order nodes FINISH
    #   Transpose the graph (reverse every edge)
    #   Pass 2: take nodes in REVERSE finish order. Each time we reach a node we have not
    #           visited yet, a DFS from it on the transpose reaches exactly one SCC.
    # Why it works: the node that finishes last lies in a "source" component of the condensation
    # DAG. Reversing the edges turns that source into a sink, so the DFS on the transpose cannot
    # leave its own component. The next unvisited node in reverse finish order is a source of
    # whatever is left, and so on.
    # The components come out in topological order of the condensation (upstream components first)
    # Runs in O(V + E): two DFS passes plus one pass over the edges to build the transpose
    def strongly_connected_components(self):
        order = self.dfs_finish_order()
        reversed_graph = self.transpose()
        visited = set()          # ONE visited set shared by every search in pass 2
        components = []
        for node in reversed(order):
            if node not in visited:
                component = []
                reversed_graph._collect_component(node, visited, component)
                components.append(component)
        return components

    def _collect_component(self, node, visited, component):
        visited.add(node)
        component.append(node)
        for neighbor in self.nodes[node]:
            if neighbor not in visited:
                self._collect_component(neighbor, visited, component)

    # Returns every stored directed edge as a list of (u, v, weight) tuples.
    # For a graph built with add_undirected_edge, each undirected edge appears TWICE this way
    # (once as (u,v,w), once as (v,u,w)) since that's how it's actually stored -- use
    # get_undirected_edges() below if you want each undirected edge counted once
    def get_edges(self):
        result = []
        for node in self.nodes:
            for neighbor, weight in self.nodes[node].items():
                result.append((node, neighbor, weight))
        return result

    # Returns a list of all edges in the graph
    # Prevents double counting (if we have <u,v> we dont need <v,u>)
    def get_undirected_edges(self):
        seen = set()
        result = []
        for (u, v, w) in self.get_edges():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            result.append((u, v, w))
        return result

    # Dijkstra's algorithm: shortest paths from source to every node, using a binary heap
    # (Python's heapq) as the priority queue.
    # Returns (dist, parent):
    #   dist[v]   = length of the shortest path from source to v (float('inf') if v is unreachable)
    #   parent[v] = the node just before v on that shortest path (None for the source and for
    #               unreachable nodes). Follow parents backwards from v to rebuild the path.
    # The idea: always finalize the closest node we have not finalized yet. Its distance can no
    # longer improve, because every other route to it must pass through a node that is at least
    # as far away, and edge weights are never negative. Then "relax" its outgoing edges: if going
    # through it gives a neighbor a shorter route than we knew, update that neighbor.
    # REQUIRES non-negative edge weights. Our add_edge/add_undirected_edge reject negative
    # weights, so that holds for every graph built here. Once negative edges are allowed (see
    # Bellman-Ford), this method silently returns WRONG answers: a finalized node is never revisited.
    # Works for directed and undirected graphs (an undirected edge is stored in both directions).
    # Instead of lowering a node's entry inside the heap (heapq cannot do that), we push a new
    # entry and skip stale ones when they come out. The heap can hold up to E entries, so the
    # running time is O((V + E) log V).
    def dijkstra(self, source, counter=None):
        if source not in self.nodes:
            raise ValueError("Illegal node entered")
        dist = {node: float('inf') for node in self.nodes}
        parent = {node: None for node in self.nodes}
        dist[source] = 0
        heap = [(0, source)]          # (distance, node), the smallest distance comes out first
        if counter is not None:
            counter.pq_ops += 1
        done = set()                  # nodes whose distance is final
        while len(heap) > 0:
            if counter is not None:
                counter.check_timeout()
            d, node = heapq.heappop(heap)
            if counter is not None:
                counter.pq_ops += 1
            if node in done:          # stale entry: we already finalized this node
                continue
            done.add(node)
            for neighbor, weight in self.nodes[node].items():
                if counter is not None:
                    counter.check_timeout()
                    counter.relaxations += 1
                if neighbor not in done and d + weight < dist[neighbor]:
                    dist[neighbor] = d + weight
                    parent[neighbor] = node
                    heapq.heappush(heap, (dist[neighbor], neighbor))
                    if counter is not None:
                        counter.pq_ops += 1
        return dist, parent

    # Convenience wrapper: returns (distance, path) for one source/target pair.
    # method can be "dijkstra" (non-negative weights only) or "bellman_ford".
    # If target is unreachable, return (inf, None). Bellman-Ford raises ValueError
    # when a negative cycle reachable from source means no shortest path exists.
    def shortest_path(self, source, target, method="dijkstra"):
        if target not in self.nodes:
            raise ValueError("Illegal node entered")
        if method == "dijkstra":
            dist, parent = self.dijkstra(source)
        elif method == "bellman_ford":
            result = self.bellman_ford(source)
            if result is None:
                raise ValueError("A negative cycle can be reached from the source")
            dist, parent = result
        else:
            raise ValueError("method must be 'dijkstra' or 'bellman_ford'")
        if dist[target] == float('inf'):
            return float('inf'), None
        path = [target]
        node = target
        while node != source:
            node = parent[node]
            if node is None:
                return float('inf'), None
            path.append(node)
        path.reverse()
        return dist[target], path

    # ------------------------------------------------------------------
    # Bellman-Ford algorithm
    # ------------------------------------------------------------------
    # Handles negative edge weights and returns None if a negative cycle can be
    # reached from source. Otherwise returns (dist, parent), like dijkstra.
    def bellman_ford(self, source, counter=None):
        if source not in self.nodes:
            raise ValueError("Illegal node entered")
        dist = {node: float('inf') for node in self.nodes}
        parent = {node: None for node in self.nodes}
        dist[source] = 0
        edges = self.get_edges()

        for _ in range(self.num_nodes - 1):
            changed = False
            for u, v, weight in edges:
                if counter is not None:
                    counter.check_timeout()
                    counter.relaxations += 1
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u
                    changed = True
            if not changed:
                break

        # One extra pass checks whether any reachable edge can still improve.
        for u, v, weight in edges:
            if counter is not None:
                counter.check_timeout()
                counter.relaxations += 1
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                return None
        return dist, parent

    # ------------------------------------------------------------------
    # Johnson's all-pairs shortest path algorithm
    # ------------------------------------------------------------------
    # Reweights a copy of the graph so Dijkstra can handle negative edges, runs
    # Dijkstra from every node, then converts the distances back to original weights.
    # Returns (dist, parent), or None if the graph has a negative cycle.
    def johnsons_apsp(self, verbose=False, counter=None, store_results=True):
        def say(message):
            if verbose:
                print(message)

        augmented = self._copy()
        s_node = max(self.nodes) + 1 if self.nodes else 0
        augmented.add_node(s_node)
        for node in self.nodes:
            augmented.add_edge(s_node, node, 0)
        say("--- 1. Added new node S (%s) with a 0-weight edge to every node ---" % s_node)
        result = augmented.bellman_ford(s_node, counter=counter)
        if result is None:
            say("--- Bellman-Ford found a negative cycle ---")
            return None
        h = result[0]
        say("--- 2. Bellman-Ford from S gives h(v) ---")
        say(h)

        reweighted = self._reweighted_copy(h)
        say("--- 3. Reweighted graph (every edge is now >= 0) ---")
        say(reweighted)

        dist = {}
        parent = {}
        for u in self.nodes:
            if counter is not None:
                counter.check_timeout()
            d_prime, source_parent = reweighted.dijkstra(u, counter=counter)
            say("--- Dijkstra from %s on the reweighted graph ---" % u)
            say(d_prime)
            if store_results:
                parent[u] = source_parent
                dist[u] = {}
                for v in self.nodes:
                    if d_prime[v] == float('inf'):
                        dist[u][v] = float('inf')
                    else:
                        dist[u][v] = d_prime[v] - h[u] + h[v]
        say("--- 4. Distances restored (D = D' - h(u) + h(v)) ---")
        return dist, parent

    # Return a NEW graph with the same nodes and edges.
    def _copy(self):
        copy = Graph()
        for u in self.nodes:
            copy.add_node(u)
        for u, v, weight in self.get_edges():
            copy.add_edge(u, v, weight)
        return copy

    # Return a NEW graph with each edge reweighted to w + h[u] - h[v].
    def _reweighted_copy(self, h):
        reweighted = Graph()
        for u in self.nodes:
            reweighted.add_node(u)
        for u, v, weight in self.get_edges():
            reweighted.add_edge(u, v, weight + h[u] - h[v])
        return reweighted

    # Adds a directed edge node1 -> node2
    # Cycles are allowed (this class used to be DAG-only; that check was removed in Week 7)
    # We will make sure that:
    # 1) Edge weights cannot be negative in this project.
    # 2) Node1 != Node2
    # 3) That the edge node1 -> node2 does not already exist
    #    (node2 -> node1 may exist: that is just a 2-cycle)
    def add_edge(self, node1, node2, weight):
        if weight < 0:
            return
        if node1 == node2:
            # print("Start and end node are the same")
            return
        self.add_node(node1)
        self.add_node(node2)
        if self.check_edge(node1, node2):
            self.nodes[node1][node2] = weight
            self.num_edges += 1

    # Adds an edge between node1 and node2 
    # We will make sure that:
    # 1) Edge weights cannot be negative in this project.
    # 2) Node1 != Node2
    # 3) That no edge already exists between the nodes
    # We do not perform the cycle check as all paths are cycles 
    # in an undirected graph
    def add_undirected_edge(self, node1, node2, weight):
        if weight < 0:
            return
        if node1 == node2:
            # print("Start and end node are the same")
            return
        self.add_node(node1)
        self.add_node(node2)
        # We do not use the cycle check as we know we are creating one
        if node2 not in self.nodes[node1]:
            self.nodes[node1][node2] = weight
            self.nodes[node2][node1] = weight
            self.num_edges += 2
        


    def __str__(self):
        result = ""
        for node in self.nodes:
            result += "Node %s: " % node
            for edge in self.nodes[node]:
                result += "{Node %s, %d} " % (edge, self.nodes[node][edge])
            result += "\n"
        return result


# Method to construct graphs for testing
# Will construct a graph that has num_nodes number of nodes and num_edges number of edges
# If num_nodes < 10, it will be set to 10
# If num_edges < num_nodes, it will be set to num_nodes
# If num_edges > num_nodes(num_nodes - 1), it will be set to num_nodes(num_nodes - 1)
# Currently the weight of an edge will be between 1 and 10, this can be modified for your purposes
def make_graph(num_nodes, num_edges):
    if num_nodes < 10:
        num_nodes = 10
    if num_edges < num_nodes:
        num_edges = num_nodes
    elif num_edges > (num_nodes * (num_nodes -1)):
        num_edges = num_nodes * (num_nodes -1)
    result = Graph()
    for i in range(num_nodes):
        result.add_node(i)
    while result.get_num_edges() < num_edges:
        node1 = random.randint(0,num_nodes-1)
        node2 = random.randint(0,num_nodes-1)
        weight = random.randint(1,10)
        result.add_edge(node1, node2, weight)
    return result

# Method to construct undirected graphs for testing
# Will construct a graph that has num_nodes number of nodes and num_edges number of edges
# If num_nodes < 10, it will be set to 10
# If num_edges < num_nodes, it will be set to num_nodes
# If num_edges > num_nodes(num_nodes - 1)/2, it will be set to num_nodes(num_nodes - 1)/2
# This number is then doubled, as every <u,v,c> edge is also <v,u,c>
def make_undirected_graph(num_nodes, num_edges):
    if num_nodes < 10:
        num_nodes = 10
    if num_edges < num_nodes:
        num_edges = num_nodes
    elif num_edges > (num_nodes * (num_nodes -1)/2):
        num_edges = num_nodes * (num_nodes -1)/2
    result = Graph()
    for i in range(num_nodes ):
        result.add_node(i)
    while result.get_num_edges() < (num_edges * 2):
        node1 = random.randint(0,num_nodes-1)
        node2 = random.randint(0,num_nodes-1)
        weight = random.randint(1,10)
        result.add_undirected_edge(node1, node2, weight)
    return result


# def main():
#     print("Welcome to our testing of graphs!")
#     print("This will randomly create a graph and add a number of specified edges to it")
#     num_nodes = int(input("Please enter the number of nodes you want in the directed graph: "))
#     num_edges = int(input("Please enter the number of edges you want in the directed graph: "))
#     test_graph = make_graph(num_nodes, num_edges)
#     # num_nodes = int(input("Please enter the number of nodes you want in the undirected graph: "))
#     # num_edges = int(input("Please enter the number of edges you want in the undirected graph: "))
#     # test_graph2 = make_undirected_graph(num_nodes, num_edges)
#     print(test_graph)
#     print(test_graph.get_component(0))
#     print(test_graph.get_component(4))
#     print(test_graph.get_component(9))
#     print("Strongly connected components:")
#     print(test_graph.strongly_connected_components())
#     print("Shortest distances from node 0 (Dijkstra):")
#     dist, parent = test_graph.dijkstra(0)
#     print(dist)

#     # print("Now that we have randomly generated graphs, let us test with reading graphs from file")
#     # in_name = "test_graph_1.txt" # Change this to read in different input files
#     # try:
#     #     in_file = open(in_name)
#     #     num_nodes = int(in_file.readline())
#     #     type_graph = in_file.readline().strip()
#     #     print(type_graph)
#     #     test_graph = Graph()
#     #     if type_graph == "UNDIRECTED":
#     #         for line in in_file:
#     #             to_add = line.split("\t")
#     #             test_graph.add_undirected_edge(int(to_add[0]), int(to_add[1]), int(to_add[2]))
#     #     if type_graph == "DIRECTED":
#     #         for line in in_file:
#     #             to_add = line.split("\t")
#     #             test_graph.add_edge(int(to_add[0]), int(to_add[1]), int(to_add[2]))
#     #     in_file.close()
#     #     print(test_graph)
#     #     print("Let us now test the component finding function. It should reveal two distinct components after our call")
#     #     print(test_graph.get_component(0))
#     #     print(test_graph.get_component(7))
#     # except IOError:
#     #     print("There was an error reading the file.")

# if __name__ == "__main__":
#     main()
