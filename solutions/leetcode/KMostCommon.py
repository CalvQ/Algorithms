'''
Problem:        K Most Common
Statement:      Given an input array of elements, return the k most common 
                elements of the list
Example: 
    Input: l=[1,1,2,2,3,5], k=2
    Output: [1,2]
Explanation:
    1 and 2 are the 2 most common elements of the list 
'''

class Solution:
    def kmostcommon(self, l: [int], k: int) -> [int]:
        d = {}
        for n in l:
            d[n] = d.get(n, 0) + 1
        out = list(zip(d.keys(), d.values()))
        out.sort(key=lambda x: x[1], reverse=True)
        return [pair[0] for pair in out[:k]]

assert sorted(Solution().kmostcommon([1,1,2,2,3,5], 2)) == sorted([1,2])
