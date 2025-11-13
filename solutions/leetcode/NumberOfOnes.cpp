/*--------------------------------------------------------------------------
Problem:            Number of 1 bits
Statement:          Given an integer, find the number of 1 bits it has.
Example:
    one_bits(23)
    ->  4
Explanation:        23 = 0b10111, which has 4 ones.
--------------------------------------------------------------------------*/

#include <iostream>

int one_bits(int num){
    int count = 0;
    int mask = 0x1;

    while(num != 0){
        if(num & mask){
            count++;
        }

        num = num >> 1;
    }

    return count;
}

int main(){
    std::cout << one_bits(23) << "\n";
}
