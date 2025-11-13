'''
Problem:            Shortest File Path
Statement:          Given an absolute pathname, return the shortest standardized
                    path.
Example:
    Input:  /usr/bin/../bin/./scripts/../
    Output: /usr/bin/
'''

class Solution:
    def shortPath(self, path):
        #Stack to store each "directory level"
        stack = []

        levels = path.split("/")

        #For each term in pathname, modify stack
        for x in levels:
            if x == "..":
                stack.pop()
            elif x == ".":
                continue
            else:
                stack.append(x)
        
        #Construct final filepath
        output = ""
        while len(stack) != 1:
            output = "/" + stack.pop() + output
        output = stack.pop() + output

        return output

path = "/usr/bin/../bin/./scripts/../"
print(Solution().shortPath(path))
