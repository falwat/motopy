from numpy import random


a = random.rand(1, 10)
b = random.rand(1, 10)
a_add_b = a + b
print(a_add_b)
abt = a @ b.T
print(abt)
ab = a * b
print(ab)
