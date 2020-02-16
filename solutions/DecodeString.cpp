/*--------------------------------------------------------------------------
Problem:            Decode String
Statement:          Given a string with a certain rule: k[string] should be
                    expanded to string k times.
Example:
    decodeString('2[a2[b]c]')
    ->  abbcabbc
--------------------------------------------------------------------------*/

#include <iostream>
#include <stack>
#include <string>

std::string decodeString(std::string input){
    /*create stack to evaluate the expression*/
    std::stack<char> stack;

    /*loop through the expression*/
    for(unsigned int i=0; i<input.length(); i++){

        /*if the character we find is a close bracket,
        * then go back until you find the matching open bracket*/
        if(input.at(i) == ']'){
            /*temp char to hold each character*/
            char temp = stack.top();
            /*string to store the resulting string in the brackets*/
            std::string eval;
            while(temp != '['){
                eval += temp;
                stack.pop();
                temp = stack.top();
            }
            /*pop the '[' off the stack*/
            stack.pop();

            /*variable to see how many copies we need*/
            int copies = stack.top() - '0';

            /*pop the number off the stack*/
            stack.pop();

            /*push the resulting string onto the original stack*/
            for(int j=0; j<copies; j++){
                for(unsigned int k=0; k<eval.length(); k++){
                    stack.push(eval.at(k));
                }
            }

        /*otherwise, just push to stack*/
        }else {
            stack.push(input.at(i));
        }
    }

    /*final string to contain everything in the stack*/
    std::string result;
    while(!stack.empty()){
        result += stack.top();
        stack.pop();
    }

    /*return string*/
    return result;
}


int main(){
    std::string input;
    std::cout << "Enter prompt: \n";
    std::getline(std::cin,input);

    std::cout << "Result is: " << decodeString(input) << "\n";
}
