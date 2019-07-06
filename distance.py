import numpy as np
def Manhattan(x1, x2):
    return np.sum(np.abs(x1-x2)[0])

def Euclidean(x1, x2):
    diff = x2 - x1
    return np.sqrt(np.dot(diff, diff))
