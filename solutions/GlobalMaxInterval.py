'''
Problem:            Global Maximum Interval
Statement:          Given an input array x, return the largest minimum interval 
                    within a size m subset of x. x is a sorted array.
Example:
    Input:  x = [2,3,5,9], m = 3
    Output: 3
Explanation:
    subset = [2,5,9]
    minimum interval = 5-2 = 3
'''

from typing import List


class Solution:
    def globalMax(self, nums: List[int], m: int) -> int:
        intervals = []
        for i in range(len(nums)-1):
            intervals.append(nums[i+1]-nums[i])

        def t_calc(t): return t[0] + t[1]

        while len(intervals) > m-1:
            gaps = [t_calc((intervals[i], intervals[i+1]))
                    for i in range(len(intervals)-1)]
            minimum = min(gaps)
            index = gaps.index(minimum)
            intervals = intervals[:index] + [minimum] + intervals[index+2:]
        return min(intervals)


maximum = Solution().globalMax([2, 3, 5, 9], 3)
assert maximum == 3
