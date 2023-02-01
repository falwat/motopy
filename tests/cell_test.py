import numpy as np


C = [[1, 2, 3], ['4', np.array([5, 6]), [7, 8, 9]]]
print(C)
# C{1} = 0;
C1 = [[None for _c in range(3)] for _r in range(3)]
print(C1)
C2 = [[None for _c in range(3)] for _r in range(2)]
print(C2)
C3 = [[[None for _c in range(4)] for _r in range(3)] for _k in range(2)]
print(C3)
