from abc import ABC, abstractmethod
from distance import Euclidean, Manhattan

class KNN(ABC):
    def __init__(self, x_data, labels, k_neighbours, distance='Euclidean'):
        self.d = self.make_distance(distance)
        self.x_data = x_data
        self.labels = labels
        self.k_neighbours = k_neighbours
    
    # calculate distance
    def make_distance(self, distance):
        d = None
        if (distance == 'Euclidean'):
            d = Euclidean()
        if (distance == 'Manhattan'):
            d = Manhattan()
        return d
    
    # return sorted distance list 
    def default_search(self, ux):
        dist = []
        for i in range(self.x_data.shape[0]):
            p_distance = self.d.distance(self.x_data[i], ux)
            dist.append((self.labels[i], p_distance))
        dist = sorted(dist, key = lambda d : d[1])
        return dist
    
    @abstractmethod
    def predict(self, ux, method=None):
        pass