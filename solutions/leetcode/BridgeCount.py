'''
Problem:            Bridge Count
Statement:          Given an undirected graph, return the number of bridges
                    within the graph.
Example:
    Input:  [[0, [1,2,3], [1, [0,2]], [2, [0,1], [3, [0,4]], [4, [3]]]
    Output: 2
Explanation:
    Total of 3 SCC's in the graph, and there are 2 bridges connecting these SCC's.
    Therefore there are 2 bridges, which would disconnect the graph if removed. 
'''

from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.visited = defaultdict(lambda: False)
        self.disc = defaultdict(lambda: float("Inf"))
        self.low = defaultdict(lambda: float("Inf"))
        self.parent = defaultdict(lambda: -1)
        self.time = 0
        self.bc = 0

    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def bridgeHelp(self, u):
        self.visited[u] = True
        self.disc[u] = self.time
        self.low[u] = self.time
        self.time += 1

        for v in self.graph[u]:
            if not self.visited[v]:
                self.parent[v] = u
                self.bridgeHelp(v)

                self.low[u] = min(self.low[u], self.low[v])

                if self.low[v] > self.disc[u]:
                    # print("Bridge between: ", u, v)
                    self.bc += 1

            elif v != self.parent[u]:
                self.low[u] = min(self.low[u], self.disc[v])

    def bridges(self):
        for u in self.graph:
            if not self.visited[u]:
                self.bridgeHelp(u)

        return self.bc


g1 = Graph()
g1.addEdge(1, 0)
g1.addEdge(0, 2)
g1.addEdge(2, 1)
g1.addEdge(0, 3)
g1.addEdge(3, 4)

print(g1.bridges())
