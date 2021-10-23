'''
Problem:            Prime Visitation
Statement:          Given an input array x and a list of integers nums, iterate
                    through each integer in nums. For each integer, flip each 
                    p_i'th term in x for each prime factor p_i of n. Return
                    the final value of x.
Example:
    Input:  x = [0,1,0,0], n = [3,4,5]
    Output: [0,0,1,1]
Explanation:
    3: [0,1,1,0]
    4: [0,0,1,1]
    5: [0,0,1,1]
'''
from typing import List


class Solution:
    def primeFactors(self, n: int) -> List[int]:
        factors = set()
        for i in range(2, n+1):
            if n % i == 0:
                n /= i
                factors.add(i)
        return factors

    def primeVisits(self, x: List[int], nums: List[int]) -> List[int]:
        d = {}
        for n in nums:
            factors = self.primeFactors(n)
            for p in factors:
                d[p] = (d.get(p, 0) + 1) % 2

        for key, val in d.items():
            if val:
                for i in range(key-1, len(x), key):
                    x[i] = not x[i]
        return x


visited = Solution().primeVisits([1, 1, 0, 0, 1, 1, 0, 1, 1, 1], [3, 4, 15])
assert visited == [1, 0, 0, 1, 0, 0, 0, 0, 1, 1]
