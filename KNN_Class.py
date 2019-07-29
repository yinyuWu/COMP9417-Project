import numpy as np
import pandas as pd
import collections
import random
import time
from BallTree import BallTree
from scipy.io import arff
from KNN import KNN
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from distance import Euclidean, Manhattan

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
        if (len(self.x_data)!=0):
            self.le = LabelEncoder()
            self.transformed_label = self.le.fit_transform(self.labels)
            #print(self.transformed_label)
            self.balltree = BallTree(self.preprocess_data(), self.d)
    
    # Helper Functions
    # ***********************************
    # combine data together
    def preprocess_data(self):
        data = np.concatenate([self.x_data, np.ones((self.x_data.shape[0],1),dtype=self.x_data.dtype)], axis=1)
        for i in range(self.x_data.shape[0]):
            data[i, -1] = self.transformed_label[i]
        return data
    
    def vote_class(self, q):
        # count frequencies of these lables then sort
        count = collections.Counter(q)
        return count.most_common(1)[0][0]
    
    def add_queue(self, q, item):

        # if item is smaller than the last element
        if (item[1] < q[self.k_neighbours-1][1]):
            q.append(item)
        else:
            for i in range(len(q)):
                if (q[i][1] <= item[1]):
                    q.insert(i, item)
                    break
        
        # if queue size is larger than knn
        if (len(q)>self.k_neighbours):
            return q[1:]
        return q
    
    def extract_knn(self, q):
        neighbour = []
        for each in q:
            predicted_v = np.array([each[0][-1]])
            neighbour.append(self.le.inverse_transform(predicted_v)[0])
        #print(len(q))
        #print(len(neighbour))
        return neighbour

    # ***********************************
    
    def BallTreeSearch(self, balltree, ux, Q):
        # if ball tree is none
        if (balltree == None):
            return

        centroid_ux_dist = self.d.distance(balltree.centroid, ux)
        first_ux_dist = Q[0][1]

        # if distance(t, B.pivot) - B.radius ≥ distance(t, Q.first) then return Q unchanged
        if (centroid_ux_dist - balltree.radius >= first_ux_dist):
            return Q

        # if current node is a leaf node
        if (balltree.right_child == None and balltree.left_child == None and centroid_ux_dist < first_ux_dist):
            return self.add_queue(Q, (balltree.data, centroid_ux_dist))


        # if current node is internal node
        else:
            child1 = None
            child2 = None
            
            if (balltree.left_child != None and balltree.right_child != None):
                if(self.d.distance(balltree.left_child.centroid, ux) >= self.d.distance(balltree.right_child.centroid, ux)):
                    child1 = balltree.right_child
                    child2 = balltree.left_child
                else:
                    child1 = balltree.left_child
                    child2 = balltree.right_child
                #print("continue searching1")
                Q = self.BallTreeSearch(child1, ux, Q)
                #print("search a further ball")
                Q = self.BallTreeSearch(child2, ux, Q)
            return Q

                        
    
    
    def predict(self, ux, method=None, distance = 'Euclidean'):
        if (self.d == None):
            print("No such distance method.")
        
        # find nearest k neibours by distance method
        if (method == "BallTree"):
            #s = time.time()
            queue = []
            for i in range(0, self.k_neighbours):
                queue.append((1, 9999999))
            neighbours = self.extract_knn(self.BallTreeSearch(self.balltree, ux, queue))
            #e = time.time()
            #print("ball tree search time: " + str(e-s))
        else:
            # default method to search knn
            dist = self.default_search(ux)
            neighbours = []
            for k in range(self.k_neighbours):
                neighbours.append(dist[k][0])
        # count frequencies of these lables then sort
        return self.vote_class(neighbours)

    

class WKNN_Class(KNN_Class):
    def __init__(self, x_data=[], labels=[], k_neighbours=7):
        super(WKNN_Class, self).__init__(x_data, labels, k_neighbours)
    
    def calc_weight(self, dist):
        if dist == 0:
            dist = 0.00000000001
        return 1/dist
    
    def extract_knn(self, q):
        neighbour = []
        w = []
        for each in q:
            predicted_v = np.array([each[0][-1]])
            weight = self.calc_weight(each[1])
            neighbour.append(self.le.inverse_transform(predicted_v)[0])
            w.append(weight)
        #print(len(q))
        #print(len(neighbour))
        return (neighbour, w)

    def predict(self, ux, method = None, distance = 'Euclidean'):
        if (self.d == None):
            print("No such distance method.")

        # find nearest k neibours by distance method
        if (method == 'BallTree'):
            queue = []
            for i in range(0, self.k_neighbours):
                queue.append((1, 9999999))
            (neighbours, weights) =  self.extract_knn(self.BallTreeSearch(self.balltree, ux, queue))
        else:
            dist = self.default_search(ux)
            neighbours = []
            weights = []
            for k in range(self.k_neighbours):
                w = self.calc_weight(dist[k][1])
                neighbours.append(dist[k][0])
                weights.append(w)
        
        # sum weights of the same label
        neighbour_dict = {}
        for i in range(len(neighbours)):
            each = neighbours[i]
            neighbour_dict[each] = neighbour_dict.get(each, 0)
            neighbour_dict[each] += weights[i]

        # get frequency of each label 
        count = collections.Counter(neighbours)

        # frequency * weight
        for key, w in neighbour_dict.items():
            freq = count.get(key,0)
            neighbour_dict[key] = w*freq

        sorted_neighbour = sorted(neighbour_dict.items(), key=lambda kv:kv[1], reverse=True)
        return sorted_neighbour[0][0]


def Test_KNN_Class(x_data, labels):
    # seperate test and training data. 10 for test set, and rest for training set
    test_size = int(x_data.shape[0]*0.4)
    x_test = x_data[test_size: , :]
    x_train = x_data[:test_size, :]
    #print("To test: " + str(test_num))
    #print("x training set shape: " + str(x_train.shape))
    y_test = labels[test_size:]
    y_train = labels[:test_size]
    knn = KNN_Class(x_train, y_train, k_neighbours=5)
    correct = 0
    start = time.time()
    for i in range(test_size):
        label_predict = knn.predict(x_test[i], method="BallTree")
        if (label_predict == y_test[i]):
            correct += 1
    end = time.time()
    return (correct/test_size, end-start)

def cross_validation(x_data, labels, knn, k_neighbours=7):
    loo = LeaveOneOut()
    predicted_error = 0
    cnt = 0
    for train_index, test_index in loo.split(x_data):
        # Split training and test data
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Set training data to knn 
        knn.d = Euclidean()
        knn.x_data = X_train
        knn.labels = y_train
        knn.k_neighbours = k_neighbours
        knn.le = LabelEncoder()
        knn.transformed_label = knn.le.fit_transform(knn.labels)
        knn.balltree = BallTree(knn.preprocess_data(), knn.d)
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
    #Test_KNN_Class(x_data, labels)

    (acc, time) = Test_KNN_Class(x_data, labels)
    print("Accuracy of knn is " + str(acc), ", time is " + str(time))



    '''
    # Cross Validation for KNN
    print("Cross Validation for normal KNN")
    for i in range(1,10):
        cross_validation(x_data, labels, KNN_Class(), i)
    # Cross Validiation for KNN Weighted
    print("Cross Validation for weighted KNN")
    for i in range(1, 10):
        cross_validation(x_data, labels, WKNN_Class(), i)
    '''
    

    


if __name__ == "__main__":
    main()