import numpy as np


s = 10
H = np.zeros((s, s))

for c in range(1, s + 1):
    for r in range(1, s + 1):
        H[r - 1:r, c - 1:c] = 1 / (r + c - 1)

print(H)

for i in np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
    print(i)
