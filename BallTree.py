import numpy as np

class BallTree():
    def __init__(self, data):
        self.data = data
        self.center = np.mean(self.data)
        self.radius = np.sqrt(np.max(np.sum((self.data - self.center)**2)))
        self.left_child = None
        self.right_child = None

    def find_left(self):
        pass
    
    def find_right(self):
        pass