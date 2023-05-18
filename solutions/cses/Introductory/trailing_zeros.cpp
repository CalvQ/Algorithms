#include <iostream>

using namespace std;

int main() {
    int n;
    cin >> n;
    int five = 0;
    int counter;
    for (int i = 5; i <= n; i += 5) {
        counter = i;
        while (counter % 5 == 0) {
            five++;
            counter /= 5;
        }
    }
    cout << five << endl;
}