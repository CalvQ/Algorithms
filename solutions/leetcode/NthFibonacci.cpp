/*--------------------------------------------------------------------------
Problem:            Nth Fibonacci
Statement:          Given a number n, print the n-th Fibonacci Number. 
Example:
    fibonacci(9)
    ->  34
Note:               This assumes that the sequence begins with 1,1,2, etc.
--------------------------------------------------------------------------*/

#include <iostream>

unsigned long fibonacci(unsigned int input, unsigned long* calculated){
    /*basis cases*/
    if(input == 1) return 1;
    if(input == 2) return 1;

    /*if numbers haven't been calculated yet, then calculate them
    * and store the result in the array*/
    if(calculated[input-1] == 0){
        calculated[input-1] = fibonacci(input-1, calculated);
    }
    if(calculated[input-2] == 0){
        calculated[input-2] = fibonacci(input-2, calculated);
    }

    /*return the previous two numbers' sum*/
    return calculated[input-1] + calculated[input-2];
}

int main(){
    int input;
    std::cout << "Enter prompt: \n";
    std::cin >> input;

    /*initialize array and initialize values*/
    unsigned long* calculated = new unsigned long[input+1];
    for(int ind=0; ind<input+1; ind++){
        calculated[ind] = 0;
    }

    std::cout << "The " << input <<"th Fibonacci Number is "
        << fibonacci(input, calculated) << "\n";

    /*deallocate to prevent memory leaks*/
    free(calculated);
}
