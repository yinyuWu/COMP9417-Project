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
    #print(filtered.shape)

    # Separate labels & x-data
    labels = filtered['price'].to_numpy()
    x_data = filtered.drop('price', axis=1)

    # TEMPORARY TEST FOR KNN NUMERIC
    # Test_KNN_Numeric(x_data, labels)

    # Cross Validation for KNN
    print("Cross Validation for normal KNN")
    for i in range(1,10):
        knn_num.cross_validation(x_data, labels, knn_num.KNN_Numeric(), i, )

def target_test():
    # standard data
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()

    # generate data
    p = 0.5
    num_samples = data.shape[0]
    num_f = data.shape[1]-1

    mean0 = tg.generate_mean(num_f)
    cov0 = tg.generate_covariance_matrix(num_f)
    target0 = tg.TargetFunction(mean0, cov0, num_f, 0)

    mean1 = tg.generate_mean(num_f)
    cov1 = tg.generate_covariance_matrix(num_f)
    target1 = tg.TargetFunction(mean1, cov1, num_f, 1)

    print('------------------------------')
    print(f'Generating data based on target function {p} probability between the two classes...')

    class_0, class_1 = tg.generate_dataset(p, num_samples, target0, target1)
    print(class_0.shape, class_1.shape)
    print(f'First 2 data points of class 0: \n{class_0[:2]}')
    print(f'First 2 data points of class 1: \n{class_1[:2]}')

    print('------------------------------')
    print('Creating data set')
    x_data, labels = tg.make_useable_dataset(class_0, 0, class_1, 1)
    print(f'First 2 data points of x_data:\n{x_data[:2]}')
    print(f'First 2 data points labels:\n{labels[:2]}')

    err = tg.calc_bayes_error(x_data, labels, target0, p, target1, (1-p))
    print(f'Bayes error rate on this dataset: {err:.8f}%')
    # err = calc_bayes_error(x_data, labels, target0, target1)
    # print(f'Estimated Bayes error rate on this dataset: {err}%')

    print('------------------------------')
    print('Using dataset with KNN...')
    tg.cross_validation(x_data, labels, knn_class.KNN_Class())

if __name__ == "__main__":
    #standard_classification_test()
    #weighted_classification_test()
    standard_numeric_test()