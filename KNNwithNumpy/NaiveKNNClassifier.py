import numpy as np
from KNN.WeightFunctions import *

class SimpleKNNClassifier:
    def __init__(self, n_neighbours, dist_metric='euclidean', weight_func=uniform):
        '''
        Set the distance metric and number of neighbours
        '''
        assert(dist_metric == 'euclidean' or dist_metric=='manhattan')
        self.n_neighbours = n_neighbours
        self.dist_metric = dist_metric
        self.weight_func = weight_func

    def fit(self, X_train, y_train):
        '''
        Assign the training set to the classifier.
        No training is done for a linear search KNN classifier.
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.n_examples = X_train.shape[0]
        self.n_features = X_train.shape[1]

    def predict_single(self, x):
        '''
        Takes an example and computes its distance to every training example.
        Then applies a weight to the k nearest neighbours based on distance.
        Then returns the label with the most total weight amongst the k nearest training examples.
        '''
        diff = self.X_train - x
        if self.dist_metric == 'euclidean':
            distances = np.sqrt(np.sum(np.square(diff, diff), axis=1))
        else:
            distances = np.sum(np.abs(diff), axis=1)
        pairs = np.stack([distances, self.y_train], axis=1) # shape: (n_examples, 2)
        pairs = pairs[pairs[:, 0].argsort(kind='stable')] # sort by distance
        weights = self.weight_func(pairs[:self.n_neighbours, 0])
        nearest = pairs[:self.n_neighbours, 1].astype(np.int64) # get label of k nearest neighbours
        return np.bincount(nearest, weights).argmax()


    def predict(self, X_test):
        '''
        Takes a test set and predicts the label for each example.
        Then returns an array of all the predicted labels.
        '''
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_single(x))
        return np.asarray(y_pred)

    def evaluate(self, X, y):
        '''
        Performs leave-one-out cross validation to calculate the accuracy on the dataset.
        '''
        correct_count = 0 # usage: count the number of correct classifications
        n_examples = X.shape[0] # number of examples in whole dataset
        for i in range(n_examples):
            # Split data into training and test set
            train_indices = np.arange(n_examples)[np.arange(n_examples)!=i]
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[i:i+1]
            y_test = y[i]
            # Fit on training set
            self.fit(X_train, y_train)
            # Predict on test set
            y_pred = self.predict(X_test)[0]
            # Compare predicted label with actual label
            if y_pred == y_test:
                correct_count += 1
        return correct_count / n_examples

if __name__ == "__main__":
    pass
