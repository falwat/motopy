from time import time
from numpy import random


tic = time()
A = random.rand(12000, 4400)
B = random.rand(12000, 4400)
print('Elapse', time() - tic, 'second.')
C = A.T * B.T
print('Elapse', time() - tic, 'second.')
