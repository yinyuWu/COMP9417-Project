import numpy as np
import pandas as pd
import collections
from scipy.io import arff
from KNN import KNN
from distance import Euclidean, Manhattan


# knn for classification

class KNN_Class(KNN):
    def __init__(self, x_data, labels, k_neighbours):
        super(KNN_Class, self).__init__(x_data, labels, k_neighbours)
    
    def predict(self, ux):
        # find nearest k neibours by Euclidean distance
        dist = []
        for i in range(self.x_data.shape[0]):
            p_distance = Euclidean(self.x_data[i], ux)
            dist.append((self.labels[i], p_distance))
        dist = sorted(dist, key = lambda d : d[1])

        neighbours = []
        for k in range(self.k_neighbours):
            neighbours.append(dist[k][0])
        # count frequencies of these lables then sort
        count = collections.Counter(neighbours)
        return count.most_common(1)[0][0]

def Test_KNN_Class(x_data, labels):
    acc = 0
    total = 10
    # seperate test and training data. 10 for test set, and rest for training set
    x_test = x_data[:10, :]
    x_train = x_data[10:, :]
    y_test = labels[:10]
    y_train = labels[10:]
    knn = KNN_Class(x_train, y_train, 7)
    for i in range(10):
        predicted_label = knn.predict(x_test[i])
        if (predicted_label == y_test[i]):
            acc+=1
    return acc/total



def main():
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    x_data = data[:, :-1]
    labels = data[:, -1]
    acc = Test_KNN_Class(x_data, labels) * 100
    print("Accuracy of kNN classification is " + str(acc) + "%")
    


if __name__ == "__main__":
    main()