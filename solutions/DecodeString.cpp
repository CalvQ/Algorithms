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
    std::stack<char> stack;
    for(unsigned int i=0; i<input.length(); i++){
        if(input.at(i) == ']'){
            char temp = stack.top();
            std::string eval;
            while(temp != '['){
                eval += temp;
                stack.pop();
                temp = stack.top();
            }
            stack.pop();
            int copies = stack.top() - '0';
            stack.pop();
            for(int j=0; j<copies; j++){
                for(unsigned int k=0; k<eval.length(); k++){
                    stack.push(eval.at(k));
                }
            }
        }else {
            stack.push(input.at(i));
        }
    }

    std::string result;
    while(!stack.empty()){
        result += stack.top();
        stack.pop();
    }

    return result;
}


int main(){
    std::string input;
    std::cout << "Enter prompt: \n";
    std::getline(std::cin,input);

    std::cout << "Result is: " << decodeString(input) << "\n";
}
