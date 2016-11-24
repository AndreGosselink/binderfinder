import numpy as np


def euklid(vec0, vec1):
    vec = np.array(vec0, float) - np.array(vec1, float)
    euk = vec * vec
    return np.sqrt(np.sum(euk))

def theta(vec0, vec1):
    dx, dy = np.array(vec0, float) - np.array(vec1, float)

    if dy > 0:
        if dx > 0:
            return 0
        else:
            return 3
    else:
        if dx > 0:
            return 1
        else:
            return 2

