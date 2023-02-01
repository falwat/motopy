import numpy as np


def func(a, b):
    s = np.sqrt(a ** 2 + b ** 2)
    return s

def func2(a):
    if a > 0:
        print('a > 0')
    else:
        disp('a <= 0')

a = 3
b = 4
s = func(a, b)
print(s)
func2(a)

