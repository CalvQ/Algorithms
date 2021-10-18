'''
Problem:            Largest Non-Adjacent Sum
Statement:          Given an input array x, return the largest possible sum of 
                    non-adjacent elements in the array. Numbers can be 0 or
                    negative.
Example:
    Input:  [5,1,1,5]
    Output: 10
Explanation:
    5+5 = 10
'''

from typing import List


class Solution:
    def largeSum(self, nums: List[int]) -> int:
        # if the array is short
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums[0], nums[1])

        # intiialize constant-size DP array
        prevMax = [nums[0], 0]
        prevMax[1] = max(nums[0], nums[1])

        # index and loop to calculate final maximum
        ind = 2
        while(ind < len(nums)):
            newMax = max(prevMax[0] + nums[ind], prevMax[1], prevMax[0])
            prevMax = [prevMax[1], newMax]
            ind += 1
        return prevMax[-1]


sum = Solution().largeSum([5, 1, 1, 5])
assert sum == 10
