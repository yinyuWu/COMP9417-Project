from abc import ABC, abstractmethod

class KNN(ABC):
    def __init__(self, x_data, labels, k_neighbours):
        self.x_data = x_data
        self.labels = labels
        self.k_neighbours = k_neighbours
    
    @abstractmethod
    def predict(self, ux):
        pass