#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    int l, r;
    for (int i = 0; i < n; ++i) {
        cin >> l >> r;
        if ((l + r) % 3 != 0) {
            cout << "NO" << endl;
        } else {
            if ((l <= 2 * r && l >= r) || (r <= 2 * l && r >= l)) {
                cout << "YES" << endl;
            } else {
                cout << "NO" << endl;
            }
        }
    }
}
