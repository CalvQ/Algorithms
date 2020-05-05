'''
Problem:            Power Function
Statement:          Write a Power Function that is O(log n)
Example:
    pow(5,3)
    ->  125
Note:               A normal, simple power function is O(n), where
                    it simply loops n times, multiplying each loop
                    Assume power is non-negative
'''

def pow(base, power):
    if power == 0:
        return 1
    
    count = 1
    result = base

    while(count**2 <= power):
        result = result**2
        count*=2

    while(count < power):
        result*=base
        count+=1

    return result

if __name__ == "__main__":
    assert pow(5,3) == 125
