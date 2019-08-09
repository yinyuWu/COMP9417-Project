import numpy as np
from KNN.WeightFunctions import *

class SimpleKNNRegressor:
    def __init__(self, n_neighbours, dist_metric='euclidean', weight_func=uniform):
        '''
        Set the distance metric and number of neighbours.
        '''
        assert(dist_metric == 'euclidean' or dist_metric=='manhattan')
        self.n_neighbours = n_neighbours
        self.dist_metric = dist_metric
        self.weight_func = weight_func

    def fit(self, X_train, y_train):
        '''
        Assign the training set to the regressor.
        No training is done for a linear search KNN regressor.
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.n_examples = X_train.shape[0]
        self.n_features = X_train.shape[1]

    def predict_single(self, x): # Takes advantage of matrix operations
        '''
        Takes an example and computes its distance to every training example.
        Then applies a weight to the k nearest neighbours based on distance.
        Then returns the sum of the weighted y values of the k nearest training examples.
        '''
        diff = self.X_train - x
        if self.dist_metric == 'euclidean':
            distances = np.sqrt(np.sum(np.square(diff, diff), axis=1))
        else:
            distances = np.sum(np.abs(diff), axis=1)
        pairs = np.stack([distances, self.y_train], axis=1) # distance and label
        pairs = pairs[pairs[:, 0].argsort()] # sort by distance
        weights = self.weight_func(pairs[:self.n_neighbours, 0]) # works for non normalised data
        weights = weights / np.sum(weights) # normalise weights
        nearest = np.sum(pairs[:self.n_neighbours, 1] * weights)
        return nearest

    def predict(self, X_test):
        '''
        Takes a test set and predicts the y value for each example.
        Then returns an array of all the predicted y values.
        '''
        y_pred = []
        for x in X_test:
            y_pred.append(self.predict_single(x))
        return np.asarray(y_pred)

    def evaluate(self, X, y):
        '''
        Performs leave-one-out cross validation to calculate the absolute mean error on the dataset.
        '''
        total_error = 0
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
            # Add the residual error to total error
            total_error += np.abs(y_test - y_pred) / y_test
        return total_error / n_examples

if __name__ == "__main__":
    pass
