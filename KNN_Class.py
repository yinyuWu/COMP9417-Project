import numpy as np
import pandas as pd
import collections
import random
from BallTree import BallTree
from scipy.io import arff
from KNN import KNN
from distance import Euclidean, Manhattan
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder


# knn for classification
'''
function knn_search is
       input: 
           t, the target point for the query
           k, the number of nearest neighbors of t to search for
           Q, max-first priority queue containing at most k points
           B, a node, or ball, in the tree
       output: 
           Q, containing the k nearest neighbors from within B
       if distance(t, B.pivot) - B.radius ≥ distance(t, Q.first) then
           return Q unchanged
       else if B is a leaf node then
           for each point p in B do
               if distance(t, p) < distance(t, Q.first) then
                   add p to Q
                   if size(Q) > k then
                       remove the furthest neighbor from Q
                   end if
               end if
           repeat
       else
           let child1 be the child node closest to t
           let child2 be the child node furthest from t
           knn_search(t, k, Q, child1)
           knn_search(t, k, Q, child2)
       end if
   end function[2]
'''

class KNN_Class(KNN):
    def __init__(self, x_data=[], labels=[], k_neighbours=7):
        super(KNN_Class, self).__init__(x_data, labels, k_neighbours)
    
    # Helper Functions
    # ***********************************
    def preprocess_data(self):
        le = LabelEncoder()
        transformed_label = le.fit_transform(self.labels)
        data = np.concatenate([self.x_data, np.ones((self.x_data.shape[0],1),dtype=self.x_data.dtype)], axis=1)
        for i in range(len(self.x_data.shape[0])):
            data[i, -1] = transformed_label[i]
        return data
    
    def vote_class(self, q):
        return q[0]
    
    def add_queue(self, q, item):
        pass
    
    def push_queue(self, q):
        pass

    # ***********************************
    
    def BallTreeSearch(self, ux):
        data = self.preprocess_data()
        Q = []
        balltree = BallTree(data)

        # if distance(t, B.pivot) - B.radius ≥ distance(t, Q.first) then return Q unchanged
        if (len(Q)>0):
            if (Euclidean(balltree.centroid, ux) - balltree.radius >= Euclidean(Q[0], ux)):
                return Q

        # if current node is a leaf node
        elif (balltree.right_child == None and balltree.left_child == None):
            for each in balltree.data:
                if (Euclidean(each, ux) < Euclidean(Q[0], ux)):
                    self.add_queue(Q, each)
                    if (len(Q) > self.k_neighbours):
                        self.push_queue(Q)
        # if current node is internal node
        else:
            child1 = None
            child2 = None
            if(Euclidean(balltree.left_child.centroid, ux) >= Euclidean(balltree.right_child.centroid, ux)):
                child1 = balltree.right_child
                child2 = balltree.left_child
            else:
                child1 = balltree.left_child
                child2 = balltree.right_child
            self.BallTreeSearch(child1)
            self.BallTreeSearch(child2)

                        
    
    
    def predict(self, ux, method=None):
        if (method == "BallTree"):
            return self.BallTreeSearch(ux)
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

class WKNN_Class(KNN):
    def __init__(self, x_data=[], labels=[], k_neighbours=7):
        super(WKNN_Class, self).__init__(x_data, labels, k_neighbours)
    
    def calc_weight(self, dist):
        if dist == 0:
            dist = 0.00000000001
        return 1/dist
    
    def predict(self, ux, method = None):
        if (method == 'BallTree'):
            return self.BallTreePredict()

        # find nearest k neibours by Euclidean distance
        dist = []
        for i in range(self.x_data.shape[0]):
            p_distance = Euclidean(self.x_data[i], ux)
            dist.append((self.labels[i], p_distance))
        dist = sorted(dist, key = lambda d : d[1])

        neighbours = []
        weights = []
        neighbour_dict = {}

        for k in range(self.k_neighbours):
            w = self.calc_weight(dist[k][1])
            neighbours.append(dist[k][0])
            weights.append(w)

        # sum weights of the same lable
        neighbour_set = set(neighbours)
        for each in neighbour_set:
            w_sum = 0
            for i in range(len(neighbours)):
                if (neighbours[i] == each):
                    w_sum += weights[i]
            neighbour_dict[each] = w_sum
        
        sorted_neighbour = sorted(neighbour_dict.items(), key=lambda kv:kv[1], reverse=True)
        return sorted_neighbour[0][0]
    
    def BallTreePredict(self):
        return 0


def Test_KNN_Class(x_data, labels):
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

def cross_validation(x_data, labels, knn, k_neighbours=7):
    loo = LeaveOneOut()
    predicted_error = 0
    cnt = 0
    for train_index, test_index in loo.split(x_data):
        # Split training and test data
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Set training data to knn 
        knn.x_data = X_train
        knn.labels = y_train
        knn.k_neighbours = k_neighbours

        # Predict value
        predicted_value = knn.predict(X_test[0])
        if (predicted_value != y_test[0]):
            predicted_error+=1
        cnt+=1
    
    # Determine the std deviation of predicted error
    print(f"Predicted error of KNN Classification: {predicted_error/cnt}") 




def main():
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    x_data = data[:, :-1]
    labels = data[:, -1]
    #print(x_data.shape)
    '''
    for i in range(20):
        if (Test_KNN_Class(x_data, labels)):
            correct += 1
            #print("Correct!")
    print("Accuracy of knn is " + str(correct/total))
    '''
    # Cross Validation for KNN
    print("Cross Validation for normal KNN")
    for i in range(1,10):
        cross_validation(x_data, labels, KNN_Class(), i)

    # Cross Validiation for KNN Weighted
    print("Cross Validation for weighted KNN")
    for i in range(1, 10):
        cross_validation(x_data, labels, WKNN_Class(), i)
    


if __name__ == "__main__":
    main()