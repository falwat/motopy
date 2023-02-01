import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a[0, 0]
m = 1
n = 1
a[m - 1, n * 2 - 1]
a[0, :]
a[:, 0]
a[0, 0:2]
a[0:2, 0]
a[0, 1:]
r = np.array([1, 3])
a[r - 1, 1]
r = np.array([1.0, 3.0])
a[np.int32(r - 1), 1]
