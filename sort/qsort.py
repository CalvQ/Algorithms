def qsort(a):
    if len(a) <= 1:
        return a
    pivot = a[0]
    pivots = [x for x in a if x == pivot]
    smaller = qsort([x for x in a if x < pivot])
    larger = qsort([x for x in a if x > pivot])
    return smaller + pivots + larger
    
