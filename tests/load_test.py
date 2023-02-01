import numpy as np
from scipy.io import loadmat


_mat = loadmat('data.mat'); a = _mat['a']; b = _mat['b']
print(a)
print(b)
a = np.loadtxt('data.dat', ndmin = 2)
print(a)
filename = 'data.dat'
a = np.loadtxt(filename, ndmin = 2)
print(a)
