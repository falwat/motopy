import numpy as np
from scipy import signal

__all__ = [
    'hanning', 'isempty', 'length'
]

def hanning(n):
    return signal.hann(n+2)[1:-1]

def isempty(x):
    if min(np.shape(x)) == 0:
        return True
    else:
        return False

def length(x):
    if np.ndim(x) == 0:
        return 1
    else:
        return max(np.shape(x))
