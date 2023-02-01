import numpy as np
from scipy.io import loadmat


# clear 
# close all
# clc 

_mat = loadmat('data.mat'); a = _mat['a']; b = _mat['b']
print(a)
print(b)
data = np.loadtxt('data.dat', ndmin = 2)
print(data)


