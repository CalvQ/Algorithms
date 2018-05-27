def insertion(a):
 for i in range(1, len(a)):
  if a[i] < a[i-1]:
   for j in range(i):
    if a[i] < a[j]:
     a[i], a[j] = a[j], a[i]
