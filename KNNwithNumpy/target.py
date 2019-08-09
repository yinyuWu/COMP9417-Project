import numpy as np
#import matplotlib.pyplot as plt
#from KNN_Class import KNN_Class, cross_validation

from scipy.stats import multivariate_normal

class TargetFunction:       # multivariate normal distributions
    def __init__(self, mean, cov, n_dimensions, class_label):
        self.mean = mean
        self.cov = cov
        assert(len(mean) == n_dimensions)
        self.n_dimensions = n_dimensions
        self.class_label = class_label

    def generate_point(self):
        return np.random.multivariate_normal(self.mean, self.cov)

    def get_probability(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

""" Generate 'n' number of samples given p prior probability of class 0, and two target functions """
def generate_dataset(p, n, target_f1, target_f2):
    class_assign = np.random.uniform(low=0,high=1,size=n)
    class_0 = []
    class_1 = []
    for c in class_assign:
        if c < p:
            class_0.append(target_f1.generate_point())
        else:
            class_1.append(target_f2.generate_point())
    return np.asarray(class_0), np.asarray(class_1)

def calc_bayes_error(x_data, labels, target_f1, p1, target_f2, p2):
    err = 0
    for x, y in zip(x_data, labels):
        # p(error|x) * p(x)
        if y == target_f1.class_label:
            # error of classification in distribution 2 * prior of distribution 2
            err += target_f2.get_probability(x) * p2 / (target_f2.get_probability(x)*p2 + target_f1.get_probability(x)*p1)
        else:
            # error of classification in distribution 1 * prior of distribution 1
            err += target_f1.get_probability(x) * p1 / (target_f2.get_probability(x)*p2 + target_f1.get_probability(x)*p1)
    return err

""" Adding labels to data & shuffling for use in KNN """
def make_useable_dataset(c1, label1, c2, label2):
    # add labels to the dataset
    data_1 = np.insert(c1, 0, label1, axis=1)
    data_2 = np.insert(c2, 0, label2, axis=1)

    # combine them into one & shuffle
    data = np.concatenate((data_1, data_2), axis=0)
    np.random.shuffle(data)
    labels = data[:,0]
    x_data = data[:,1:]
    return x_data, labels

def generate_mean(n):
    return np.random.uniform(0,0.5,n)

def generate_covariance_matrix(n):
    r = np.random.uniform(0,0.1,(n,n))
    matrix = np.dot(r, r.transpose())
    return matrix

#### functions to use in other files for easier use
def create_target_function(num_features, class_label):
    mean = generate_mean(num_features)
    cov = generate_covariance_matrix(num_features)
    return TargetFunction(mean, cov, num_features, class_label)

def generate_data_with_labels(p, n, target0, target1):
    class_0, class_1 = generate_dataset(p, n, target0, target1)
    return make_useable_dataset(class_0, target0.class_label, class_1, target1.class_label)

if __name__=='__main__':
    np.random.seed(201923)
    p = 0.5
    num_samples = 10000
    num_f = 30

    mean0 = generate_mean(num_f)
    cov0 = generate_covariance_matrix(num_f)
    target0 = TargetFunction(mean0, cov0, num_f, 0)

    mean1 = generate_mean(num_f)
    cov1 = generate_covariance_matrix(num_f)
    target1 = TargetFunction(mean1, cov1, num_f, 1)

    print('------------------------------')
    print(f'Generating data based on target function {p} probability between the two classes...')

    class_0, class_1 = generate_dataset(p, num_samples, target0, target1)
    print(class_0.shape, class_1.shape)
    print(f'First 2 data points of class 0: \n{class_0[:2]}')
    print(f'First 2 data points of class 1: \n{class_1[:2]}')

    print('------------------------------')
    print('Creating data set')
    x_data, labels = make_useable_dataset(class_0, 0, class_1, 1)
    print(f'First 2 data points of x_data:\n{x_data[:2]}')
    print(f'First 2 data points labels:\n{labels[:2]}')

    err = calc_bayes_error(x_data, labels, target0, p, target1, (1-p))
    print(f'Bayes error rate on this dataset: {err*100:.20f}%')
    # err = calc_bayes_error(x_data, labels, target0, target1)
    # print(f'Estimated Bayes error rate on this dataset: {err}%')

    print('------------------------------')
    print('Using dataset with KNN...')
    cross_validation(x_data, labels, KNN_Class())
