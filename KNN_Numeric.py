import numpy as np
from scipy.io import arff
import pandas as pd

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

def remove_missing(data):
    return data.dropna(how='any').to_numpy()   

def main():
    data_set = arff.loadarff('autos.arff')
    data = pd.DataFrame(data_set[0])
    filtered = remove_missing(data)
    #Manhattan_Test(data)
    #Euclidean_Test(data)

if __name__ == "__main__":
    main()