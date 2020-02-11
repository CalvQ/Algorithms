'''
Breadth First Search
Identifies the shortest graph-traversal 
Searches by steps
'''
def bfs(graph, root, target):
    # Queue of visited nodes
    visited = [False] * (len(graph))
    
    # List to backtrace the path
    backtrace = [0] * (len(graph))

    # Queue for the traversal
    queue = []

    # Append the first element
    queue.append(s)
    visited[s] = True

    # While the queue isn't empty,
    # we continue searching
    while queue:
        node = queue.pop(0)
        
        for i in graph[node]:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True

                backtrace[i] = node
            if visited[target] == True:
                queue.clear()
                break

    # Print out backtrace
    print(target)
    while not backtrace[target] == root:
        print(backtrace[target])
        target = backtrace[target]
