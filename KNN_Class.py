import numpy as np
import pandas as pd
import collections
import random
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


def cross_validation(x_data, labels):
    test_num = random.randint(0, x_data.shape[0])
    # seperate test and training data. 10 for test set, and rest for training set
    x_test = x_data[test_num, :]
    x_train = np.concatenate((x_data[:test_num, :], x_data[test_num+1:, :]), axis = 0)
    #print("To test: " + str(test_num))
    #print("x training set shape: " + str(x_train.shape))
    y_test = labels[test_num]
    y_train = np.concatenate((labels[:test_num], labels[test_num+1:]),axis = 0)
    knn = KNN_Class(x_train, y_train, 7)
    label_predict = knn.predict(x_test)
    return label_predict == y_test



def main():
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    x_data = data[:, :-1]
    labels = data[:, -1]

    correct = 0
    total = 20
    #print(x_data.shape)
    for i in range(20):
        if (cross_validation(x_data, labels)):
            correct += 1
            #print("Correct!")
    print("Accuracy of knn is " + str(correct/total))
    


if __name__ == "__main__":
    main()