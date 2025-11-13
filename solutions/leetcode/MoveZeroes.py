'''
Problem:            Move Zeroes
Statement:          Given an array, move all zeroes to the end of the array.
Example:
    Input:  [0,1,0,3,12]
    Output: [1,3,12,0,0]
Restrictions:       You must do this in-place without making a copy of the 
                    array.
                    Minimize the total number of operations.
'''

class Solution:
  def moveZeros(self, nums):
    head = swap = 0
    while head < len(nums):
        #Edge case for leading non-zero elements
        if not nums[swap] == 0:
            head += 1
            swap += 1
            continue

        #Identify earliest non-zero element to be moved
        while head < len(nums) and nums[head] == 0:
            head += 1
        
        #If head is length, we've reached the end
        if head == len(nums):
            return
        
        #Swap the two elements
        temp = nums[swap]
        nums[swap] = nums[head]
        nums[head] = temp

        head += 1
        swap += 1

nums = [0, 0, 0, 2, 0, 1, 3, 4, 0, 0]
Solution().moveZeros(nums)
print(nums) 