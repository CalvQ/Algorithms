/*--------------------------------------------------------------------------
Problem:            Find Single
Statement:          Given an array full of pairs of numbers except a single
                     element, find that element.
Example:
    nums = [1, 1, 3, 4, 4, 5, 6, 5, 6]
    findSingle(nums);
    ->  3
--------------------------------------------------------------------------*/
import java.util.Scanner;

public class FindSingle{
    static int findSingle(int[] nums){
        int result = 0;
        for(int num : nums){
            result ^= num;
        }

        return result;
    }

    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        String[] input = sc.nextLine().split(" ");

        int[] nums = new int[input.length];

        for(int ind = 0; ind < input.length; ind++){
            nums[ind] = Integer.parseInt(input[ind]);
        }

        sc.close();

        System.out.println(findSingle(nums));
    }
}
