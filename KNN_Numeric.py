import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import LeaveOneOut
from scaler import MinMaxScaler
from distance import Manhattan, Euclidean
from KNN import KNN

# knn for numeric prediction

class KNN_Numeric(KNN):
    def __init__(self, x_data=[], labels=[], k_neighbours=7):
        super(KNN_Numeric, self).__init__(x_data, labels, k_neighbours)
    
    def predict_value(self, neighbours):
        return np.mean(neighbours)

    def predict(self, ux, method=None, distance = 'Euclidean'):
        dist = self.default_search(ux)
        neighbours = []
        for k in range(self.k_neighbours):
            neighbours.append(dist[k][self.LABEL_INDEX])
        return self.predict_value(neighbours)


class WKNN_Numeric(KNN_Numeric):
    def __init__(self, x_data=[], labels=[], k_neighbours=7):
        super(WKNN_Numeric, self).__init__(x_data, labels, k_neighbours)
    
    def predict_value(self, neighbours):
        # Calculate weighted sum average
        total_value = 0
        total_weight = 0
        for i in range(len(neighbours)):
            weight = self.calc_weight(neighbours[i][self.DISTANCE_INDEX])
            total_weight += weight 
            total_value += (weight*neighbours[i][self.LABEL_INDEX])
        return total_value/total_weight     

    def calc_weight(self, dist=1):  # Temporary formula for weight
        if dist == 0:
            dist = 0.00000000001
        return 1/dist

    def predict(self, ux):
        dist = self.default_search(ux)
        neighbours = []
        for k in range(self.k_neighbours):
            neighbours.append(dist[k])
        return self.predict_value(neighbours)

def Test_KNN_Numeric(x_data, labels):
    # Scale numeric features so they are between 0-1
    scaler = MinMaxScaler()
    scaled_x_data = scaler.fit_transform(x_data)

    x_test = scaled_x_data[0]       # first entry is test data
    y_test = labels[0]
    x_train = scaled_x_data[1:]     # the rest is training
    y_train = labels[1:]

    knn = KNN_Numeric(x_train, y_train, 7)
    print(f'Predicted Price: {knn.predict(x_test)}')
    print(f'Actual Price: {y_test}')

def cross_validation(x_data, labels, knn, k_neighbours=7, distance = Euclidean()):
    # Scale numeric features so they are between 0-1
    scaler = MinMaxScaler()
    scaled_x_data = scaler.fit_transform(x_data)
    knn.d = distance
    # Leave One Out Cross Validation
    loo = LeaveOneOut()
    predicted_error = []
    for train_index, test_index in loo.split(scaled_x_data):
        # Split training and test data
        X_train, X_test = scaled_x_data[train_index], scaled_x_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Set training data to knn 
        knn.x_data = X_train
        knn.labels = y_train
        knn.k_neighbours = k_neighbours
        
        # Predict value
        predicted_value = knn.predict(X_test[0])
        # Store difference between predicted value and actual value in array 
        predicted_error.append(predicted_value - y_test[0])
    
    # Determine the std deviation of predicted error
    print(f"Std deviation of predicted error of KNN: {np.std(predicted_error)}")
    return  np.std(predicted_error)

""" Args: data, list<String>
    Creates new column for each label (label name + '-numeric') that encodes categorical labels to integers 
"""
def convert_to_numeric(data, labels):
    for label in labels:
        data[label] = data[label].astype('category')
        numeric_label = label + "-numeric"
        data[numeric_label] = data[label].cat.codes 
    return data

def main():
    # Load data from autos.aff
    data_set = arff.loadarff('autos.arff')
    data = pd.DataFrame(data_set[0])    

    # Make new numeric labels for each of these categorical labels 
    data = convert_to_numeric(data, ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 
                'fuel-system'])
    
    # Remove the old categorical labels 
    data.drop(['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 
                'fuel-system', 'normalized-losses'], axis=1, inplace=True)

    # Remove missing values
    filtered = data.dropna()
    #print(data)
    print(filtered.shape)

    # Separate labels & x-data
    labels = filtered['price'].to_numpy()
    x_data = filtered.drop('price', axis=1)

    # TEMPORARY TEST FOR KNN NUMERIC
    # Test_KNN_Numeric(x_data, labels)

    # Cross Validation for KNN
    '''
    print("Cross Validation for normal KNN")
    for i in range(1,10):
        cross_validation(x_data, labels, KNN_Numeric(), i)
    
    '''
    # Cross Validiation for KNN Weighted
    print("Cross Validation for weighted KNN")
    for i in range(1, 10):
        cross_validation(x_data, labels, WKNN_Numeric(), i)
    

if __name__ == "__main__":
    main()