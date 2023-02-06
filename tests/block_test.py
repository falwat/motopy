import numpy as np


A = np.zeros((1, 2))
B = np.ones((1, 3))
C = np.zeros((2, 1))
D = np.ones((2, 4))
print(np.block([A, B]))
print(np.block([C, D]))
print(np.block([[A, B], [C, D]]))



