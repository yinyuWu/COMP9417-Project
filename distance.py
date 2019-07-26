import numpy as np

class Euclidean():
    def __init__(self):
        self.method = 'Euclidean'

    def distance(self, x1, x2):
        diff = x2 - x1
        return np.sqrt(np.dot(diff, diff))
    

class Manhattan():
    def __init__(self):
        self.method = 'Manhattan'

    def distance(self, x1, x2):
        return np.sum(np.abs(x1-x2)[0])
