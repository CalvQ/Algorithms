#include <iostream>
using namespace std;

#define ll long long

int main() {
    int n;
    cin >> n;
    ll total = 0, input = 0;
    cin >> input;
    // for each incoming number...
    for (int i = 1; i < n; ++i) {
        int j;
        cin >> j;
        if (input > j) {
            total += input - j;
        } else {
            input = j;
        }
    }
    cout << total;
}
