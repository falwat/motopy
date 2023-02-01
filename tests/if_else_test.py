import numpy as np
from numpy import random


limit = 0.75
A = random.rand(10, 1)

if any(A > limit):
    disp('There is at least one value above the limit.')
else:
    print('All values are below the limit.')
# %%
x = 10
if x != 0:
    print('Nonzero value')
# %%
x = 10
minVal = 2
maxVal = 6

if (x >= minVal) and (x <= maxVal):
    disp('Value within specified range.')
elif (x > maxVal):
    print('Value exceeds maximum value.')
else:
    disp('Value is below minimum value.')
# %%
nrows = 4
ncols = 6
A = np.ones((nrows, ncols))

for c in range(1, ncols + 1):
    for r in range(1, nrows + 1):
        if r == c:
            A[r - 1, c - 1] = 2
        elif np.abs(r - c) == 1:
            A[r - 1, c - 1] = -1
        else:
            A[r - 1, c - 1] = 0
