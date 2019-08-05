import KNN_Class as knn_class
import KNN_Numeric as knn_num
import target as tg
import pandas as pd
from distance import Euclidean, Manhattan
from scipy.io import arff


def standard_classification_test():
    # preprocess
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    x_data = data[:, :-1]
    labels = data[:, -1]

    # use cross validation to test knn classification by standard UCI data ionosphere
    # use of cross validation: cross_validation(x_data, labels, knn, k_neighbours, method, distance)
    # method: BallTree or None. Distance: Manhattan distance or Eucilidean distance
    print("Cross validation for KNN classification")
    for i in range(1, 8):
        knn = knn_class.KNN_Class()
        knn_class.cross_validation(x_data, labels, knn, i)

def weighted_classification_test():
    # preprocess
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    x_data = data[:, :-1]
    labels = data[:, -1]

    # use cross validation to test weighted knn classification by standard UCI data ionosphere
    # use of cross validation: cross_validation(x_data, labels, knn, k_neighbours, method, distance)
    # method: BallTree or None. Distance: Manhattan distance or Eucilidean distance
    print("Cross validation for weighterd KNN classification")
    for i in range(1, 8):
        wknn = knn_class.WKNN_Class()
        knn_class.cross_validation(x_data, labels, wknn, i)

def standard_numeric_test():
    # Load data from autos.aff
    data_set = arff.loadarff('autos.arff')
    data = pd.DataFrame(data_set[0])
    
    # Remove any categorical labels (Note: temporary as we can encode these categorical labels later)
    data.drop(['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 
                'fuel-system'], axis=1, inplace=True)

    # Remove missing values
    filtered = data.dropna()

    # Separate labels & x-data
    labels = filtered['price'].to_numpy()
    x_data = filtered.drop('price', axis=1)

    # Cross Validation for KNN
    print("Cross Validation for normal KNN")
    for i in range(1,10):
        knn_num.cross_validation(x_data, labels, knn_num.KNN_Numeric(), i)

def weighted_numeric_test():
    # Load data from autos.aff
    data_set = arff.loadarff('autos.arff')
    data = pd.DataFrame(data_set[0])
    
    # Remove any categorical labels (Note: temporary as we can encode these categorical labels later)
    data.drop(['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 
                'fuel-system'], axis=1, inplace=True)

    # Remove missing values
    filtered = data.dropna()

    # Separate labels & x-data
    labels = filtered['price'].to_numpy()
    x_data = filtered.drop('price', axis=1)

    # Cross Validation for KNN
    print("Cross Validation for numeric weighted KNN")
    for i in range(1,10):
        knn_num.cross_validation(x_data, labels, knn_num.WKNN_Numeric(), i)

def target_test():
    # standard data
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()

    # set known parameters
    p = 0.5
    num_samples = data.shape[0]
    num_f = data.shape[1]-1
    
    # create target functions based on number of features (dimension)
    target0 = tg.create_target_function(num_f, 0)
    target1 = tg.create_target_function(num_f, 1)
    
    # create dataset based on target functions with p probability for target0
    x_data, labels = tg.generate_data_with_labels(p, num_samples, target0, target1)
    print(x_data[:2], labels[:2])

    err = tg.calc_bayes_error(x_data, labels, target0, p, target1, (1-p))
    print(f'Bayes error rate on this dataset: {err:.8f}%')

    print('------------------------------')
    print('Using dataset with KNN...')
    tg.cross_validation(x_data, labels, knn_class.KNN_Class())

if __name__ == "__main__":
    #standard_classification_test()
    weighted_classification_test()
    #standard_numeric_test()
    #weighted_numeric_test()
    #target_test()