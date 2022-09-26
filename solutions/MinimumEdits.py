'''
Problem:            Minimum Edits
Statement:          Given an input start and goal, return the minimum number of edits needed to change
                    the start string into the goal string. Allowed edits are add, remove, and substitute.
Example:
    Input:  "test","test"
    Output: 0
Example:
    Input:  "test","wow"
    Output: 4
Explanation:
    "test" -> "est" -> "wst" -> "wot" -> "wow"
'''


class Solution:
    def minEdits(self, start: str, goal: str) -> int:
        if not start and goal:
            return self.minEdits(goal[0], goal) + 1
        if start and not goal:
            return self.minEdits(start[1:], goal) + 1
        if not start and not goal:
            return 0
        if start[:1] == goal[:1]:
            return self.minEdits(start[1:], goal[1:])
        return min(self.minEdits(goal[0] + start, goal),
                   self.minEdits(goal[0] + start[1:], goal),
                   self.minEdits(start[1:], goal)) + 1


testString = Solution().minEdits("test", "test")
testDoneeShush = Solution().minEdits("donee", "shush")
assert testString == 0
assert testDoneeShush == 5
