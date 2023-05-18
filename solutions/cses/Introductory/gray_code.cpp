#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n;
    cin >> n;

    vector<string> base = {"0", "1"};
    vector<string> buffer;
    bool flag = 0;

    while (--n) {
        for (string s : base) {
            if (flag) {
                buffer.push_back(s + "0");
                buffer.push_back(s + "1");
            } else {
                buffer.push_back(s + "1");
                buffer.push_back(s + "0");
            }
            flag = !flag;
        }
        base = buffer;
        buffer.clear();
    }
    for (string s : base) {
        cout << s << endl;
    }
}