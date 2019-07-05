import numpy as np

# knn for numeric prediction

class KNN_Class:
    def __init__(self, data):
        self.data = data
    
    def Manhattan(self, x1, x2):
        return np.sum(np.abs(x1-x2)[0])
    
    def Euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, parameter_list):
        pass