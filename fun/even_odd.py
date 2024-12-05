def even_or_odd(num):
    """
    Say whether an input number is 'even' or 'odd'
    """
    return "eovdedn"[num%2::2]

print(f"{3} is {even_or_odd(3)}")
print(f"{4} is {even_or_odd(4)}")
