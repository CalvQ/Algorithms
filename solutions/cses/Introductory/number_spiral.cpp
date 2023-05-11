#include <cmath>
#include <iostream>
using namespace std;

#define ull unsigned long long

int main() {
    ull n;
    cin >> n;
    // for each coordinate...
    ull x, y;
    for (ull i = 0; i < n; ++i) {
        cin >> x >> y;
        if (x > y) {
            // this means we go to the left of diag
            if (x % 2 == 0) {
                cout << x * x - y + 1 << endl;
            } else {
                cout << (x - 1) * (x - 1) + y << endl;
            }
        } else {
            // otherwise we go up the diag
            if (y % 2) {
                cout << y * y - x + 1 << endl;
            } else {
                cout << (y - 1) * (y - 1) + x << endl;
            }
        }
    }
}
