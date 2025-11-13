/*--------------------------------------------------------------------------
Problem:            Array Sum
Statement:          Given an array of numbers and a number k, return whether
                     any two numbers in the list can add up to k.
Example:
    nums = [10, 15, 3, 7]
    arraySum(nums, 17);
    ->  True
--------------------------------------------------------------------------*/

import java.util.HashSet;

class Solution {
    static boolean arraySum(int[] nums, int k){
        boolean found = false;

        //Contain found numbers in hashset
        HashSet<Integer> numbers = new HashSet<>();
        
        //Iterate through given array
        for(int x : nums){
            if(numbers.contains(k-x)) found = true;
            numbers.add(x);
        }

        return found;
    }

    public static void main(String[] args) {
        int[] nums = {10, 15, 3, 7};
        System.out.println(arraySum(nums, 17));
    }
}