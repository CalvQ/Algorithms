#include <cmath>
#include <iostream>
using namespace std;

#define lli long long int

int main() {
    int n;
    cin >> n;
    lli mod = (lli)pow(10, 9) + 7;
    lli output = 1;
    while (n--) {
        output *= 2;
        output %= mod;
    }
    cout << output << endl;
}