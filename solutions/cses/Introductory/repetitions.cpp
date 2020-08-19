#include <iostream>
using namespace std;

#define ll long long

int main(){
    string s;
    cin >> s;
    int ans, c=0;
    char i = 'A';
    for(char d : s){
        if(d == i){
            ++c;
            ans = max(c,ans);
        }else {
            i=d;
            c=1;
        }
    }
    cout << ans;
}
