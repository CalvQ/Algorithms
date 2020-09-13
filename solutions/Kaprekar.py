'''
Problem:            Kaprekar's Constant
Statement:          Given an input x, return the number of steps needed to reach
                    the constnant 6174
Example:
    Input:  1234
    Output: 3
Explanation:
    4321 - 1234 = 3087
    8730 - 0378 = 8352
    8532 - 2358 = 6174
'''

class Solution:
    def kaprekar(self, n: int) -> int:
        count = 0
        while n != 6174:
            num1 = int("".join(sorted(str(n))))
            num2 = int("".join(sorted(str(n), reverse = True)))
            n = num2 - num1
            count += 1
        return count

steps = Solution().kaprekar(1234)
assert steps == 3
