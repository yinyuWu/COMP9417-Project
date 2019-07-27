from abc import ABC, abstractmethod
from distance import Euclidean, Manhattan
from sklearn.preprocessing import LabelEncoder
from BallTree import BallTree
import numpy as np

class KNN(ABC):
    def __init__(self, x_data, labels, k_neighbours, distance='Euclidean'):
        if (x_data != []):
            self.d = self.make_distance(distance)
            self.x_data = x_data
            self.labels = labels
            self.k_neighbours = k_neighbours
            self.le = LabelEncoder()
            self.transformed_label = self.le.fit_transform(self.labels)
            self.balltree = BallTree(self.preprocess_data(), self.d)
    
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

    # combine data together
    def preprocess_data(self):
        data = np.concatenate([self.x_data, np.ones((self.x_data.shape[0],1),dtype=self.x_data.dtype)], axis=1)
        for i in range(self.x_data.shape[0]):
            data[i, -1] = self.transformed_label[i]
        return data
    
    @abstractmethod
    def predict(self, ux, method=None):
        pass