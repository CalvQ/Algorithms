'''
Tarjan's Algorithm
Identifies the strongly connected components in a directed graph
Searches using lowlinks and recursion
'''

from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.index = defaultdict(int)
        self.lowlink = defaultdict(int)
        self.i = 0
        self.stack = []
        self.sccList = []

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def tarjan(self):
        for v in self.graph:
            if not v in self.index:
                self.strongconnect(v)

        return self.sccList

    def strongconnect(self, v):
        print(v)
        self.index[v] = self.i
        self.lowlink[v] = self.i
        self.i += 1
        self.stack.append(v)
        for u in self.graph[v]:
            if not u in self.index:
                self.strongconnect(u)
                self.lowlink[v] = min(self.lowlink[u], self.lowlink[v])
            elif u in self.stack:
                self.lowlink[v] = min(self.index[u], self.lowlink[v])

        if self.lowlink[v] == self.index[v]:
            scc = []
            w = self.stack.pop(len(self.stack)-1)
            scc.append(w)
            while v != w:
                w = self.stack.pop(len(self.stack)-1)
                scc.append(w)
            self.sccList.append(scc)


g = Graph()
g.addEdge(1, 0)
g.addEdge(0, 2)
g.addEdge(2, 1)
g.addEdge(3, 2)
g.addEdge(3, 1)
g.addEdge(3, 4)
g.addEdge(4, 3)
g.addEdge(4, 5)
g.addEdge(5, 1)
g.addEdge(5, 6)
g.addEdge(6, 5)
g.addEdge(7, 6)
g.addEdge(7, 7)
g.addEdge(7, 4)

print(g.tarjan())
