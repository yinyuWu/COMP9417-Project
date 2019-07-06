import numpy as np
from scipy.io import arff
import pandas as pd
from distance import Manhattan, Euclidean

# knn for classification

class KNN_Class:
    def __init__(self, data, k_neighbours):
        self.x_data = data[:, :-1]
        self.lables = data[:, -1]
        self.k_neighbours = k_neighbours
    
    def predict(self, ux):
        # find nearest k neibours by Euclidean distance
        dist = np.zeros((len(self.x_data), 2))
        for i in range(self.x_data.shape[0]):
            p_distance = Euclidean(self.x_data[i], ux)
            dist[i, 0] = p_distance
            dist[i, 1] = self.lables[i]
        dist = dist[dist[:,0].argsort()]
        y = dist[:self.k_neighbours, 1]

        # count frequencies of these lables then sort
        lable, cnt = np.unique(y, return_counts=True)
        lable_dict = dict(zip(lable, cnt))
        sorted_lable_dict = sorted(lable_dict.items(), key = lambda kv : kv[1], reverse = True)
        return list(sorted_lable_dict.keys())[0]


'''
def Manhattan_Test(data):
    test_data = data[:2, :-1]
    x1 = test_data[:1, :]
    x2 = test_data[1:2, :]
    knn = KNN_Class(data,1)
    distance = knn.Manhattan(x1, x2)
    print(distance)

def Euclidean_Test(data):
    test_data = data[:2, :-1]
    x1 = test_data[:1, :]
    x2 = test_data[1:2, :]
    knn = KNN_Class(data,1)
    distance = knn.Euclidean(x1, x2)
    print(distance)
    print("norm is " + str(np.linalg.norm(x1-x2)))
'''

def main():
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    #Manhattan_Test(data)
    #Euclidean_Test(data)


if __name__ == "__main__":
    main()