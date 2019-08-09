from heapq import heappush, heappop
import numpy as np

def euclidean(x1, x2):
    diff = x2 - x1
    return np.sqrt(np.dot(diff, diff))

def manhattan(x1, x2):
    return np.sum(np.abs(x2 - x1))

class TestKNNClassifier:
    def __init__(self, n_neighbours, dist_metric='euclidean'):
        assert(dist_metric == 'euclidean' or dist_metric=='manhattan')
        self.n_neighbours = n_neighbours
        if dist_metric == 'euclidean':
            self.dist_metric = euclidean
        else:
            self.dist_metric = manhattan

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_single1(self, x):
        '''
        Uses a for-loop to go through all the training examples
        and calculates its distance from the test example. Appends
        the result and its corresponding label as a pair into a list.
        Then sorts the list by the distance and returns the most common
        label from the first k elements in the list. Does not use numpy.
        '''
        n = self.X_train.shape[0] # number of training examples
        pairs = []
        for i in range(n):
            pairs.append((self.dist_metric(self.X_train[i], x), self.y_train[i]))
        pairs.sort(key = lambda pair: pair[0])
        nearest = [pair[1] for pair in pairs[:self.n_neighbours]] # get labels of nearest neighbours
        return max(set(nearest), key=nearest.count)

    def predict_single2(self, x):
        '''
        Uses a for-loop to go through all the training examples
        and calculates its distance from the test example. Pushes
        the result ant its corresponding label into a min-heap.
        Then pops k elements from the heap and returns the most
        common label among them. Does not use numpy.
        '''
        n = self.X_train.shape[0] # number of training examples
        pairs = []
        for i in range(n):
            heappush(pairs, (self.dist_metric(self.X_train[i], x), self.y_train[i]))
        nearest = []
        for i in range(self.n_neighbours):
            nearest.append(heappop(pairs)[1])
        return max(set(nearest), key=nearest.count)

    def predict_single3(self, x):
        '''
        Uses numpy matrix operations to calculate the distance from
        the test example to all training examples. Then uses the
        stack operation to pair the distances with their label into
        a new numpy array. Then sorts the array by the distance and
        returns the most common label from the first k elements in the
        the array.
        '''
        diff = self.X_train - x
        distances = np.sqrt(np.sum(np.square(diff, diff), axis=1))
        pairs = np.stack([distances, self.y_train], axis=1) # distance and label
        pairs = pairs[pairs[:, 0].argsort()] # sort by distance
        nearest = pairs[:self.n_neighbours, 1].astype('int64') # label of nearest neighbours
        return np.bincount(nearest).argmax()

    def predict_single4(self, x): # Takes advantage of matrix operations and uses heap
        '''
        The same as the third approach but instead of sorting, it uses a
        for-loop to push all distance/label pairs onto a min-heap.
        Then pops k elements from the heap and returns the most
        common label among them.
        '''
        diff = self.X_train - x
        distances = np.sqrt(np.sum(np.square(diff, diff), axis=1))
        pairs = np.stack([distances, self.y_train], axis=1) # distance and label
        pairs = pairs.tolist() # convert to list for heap ops
        nearest = []
        ordered = []
        for pair in pairs: # have to push all pairs to a heap
            heappush(ordered, pair)
        for i in range(self.n_neighbours): # and then pop them
            nearest.append(int(heappop(ordered)[1]))
        return max(set(nearest), key=nearest.count)

    def predict1(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_single1(x))
        return np.asarray(y_pred)

    def predict2(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_single2(x))
        return np.asarray(y_pred)

    def predict3(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_single3(x))
        return np.asarray(y_pred)

    def predict4(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_single4(x))
        return np.asarray(y_pred)

if __name__ == "__main__":
    pass
