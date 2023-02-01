import numpy as np


X = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 4]])
k = np.nonzero(X)
print(k)

v = np.arange(1, 11)
k = np.nonzero(v > 5)[0]
print(k)
