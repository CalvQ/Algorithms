import numpy as np

array = np.arange(100)


def binary_search(array, target):
    low = 0
    high = len(array)-1
    half = (low + high) // 2
    while array[half] != target:
        if low > high:
            return -1
        if array[half] < target:
            low = half+1
        else:
            high = half-1
        half = (low + high) // 2

    return half


for num in np.arange(100):
    assert binary_search(array, num) == num

assert binary_search(array, -20) == -1

print("All tests passed!")
