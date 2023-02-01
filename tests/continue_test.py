import numpy as np


for n in range(1, 51):
    if np.mod(n, 7) != 0:
        continue
    print(''.join(['Divisible by 7: ', str(n)]))
