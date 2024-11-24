#include <iostream>
using namespace std;

#define ll long long

int main() {
    int n;
    cin >> n;
    ll nums[n];
    int stick;
    int counter = 0;
    while (counter < n) {
        cin >> stick;
        nums[counter] = stick;
        counter++;
    }

    sort(nums, nums + n);
    ll target = nums[n / 2];
    ll total = 0;
    for (ll num : nums) {
        total += abs(num - target);
    }
    cout << total;
}
