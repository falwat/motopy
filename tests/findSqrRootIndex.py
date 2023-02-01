import numpy as np


def findSqrRootIndex(target, arrayToSearch):
    idx = np.NaN
    if target < 0:
        return idx
    for idx in range(1, max(np.shape(arrayToSearch)) + 1):
        if arrayToSearch[idx - 1] == np.sqrt(target):
            return idx
    return idx

