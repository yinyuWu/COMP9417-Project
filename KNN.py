from abc import ABC, abstractmethod
from distance import Euclidean, Manhattan

class KNN(ABC):
    def __init__(self, x_data, labels, k_neighbours, distance='Euclidean'):
        self.d = self.make_distance(distance)
        self.x_data = x_data
        self.labels = labels
        self.k_neighbours = k_neighbours
    
    def make_distance(self, distance):
        d = None
        if (distance == 'Euclidean'):
            d = Euclidean()
        if (distance == 'Manhattan'):
            d = Manhattan()
        return d
    
    @abstractmethod
    def predict(self, ux, method=None):
        pass