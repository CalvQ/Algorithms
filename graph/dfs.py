def dfs(graph, root, visitor):
    visited = []
    def dfs_walk(node):
        visited.append(node)
        visitor(node)
        for succ in graph[node]:
            if succ not in visited:
                dfs_walk(succ)
    dfs_walk(root)
