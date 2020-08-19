#include <iostream>
using namespace std;

#define ll long long

int main(){
    int n;
    cin >> n;
    ll total=0, l=0;
    cin >> l;
    for(int i=1; i<n; ++i){
        int j;
        cin >> j;
        if(l>j){
            total += l-j;
        }else {
            l = j;
        }
    }
    cout << total;
}
