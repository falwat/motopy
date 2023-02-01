from numpy import random


limit = 0.8
s = 0

while 1:
    tmp = random.rand(1)
    print(tmp)
    if tmp > limit:
        break
    s = s + tmp
