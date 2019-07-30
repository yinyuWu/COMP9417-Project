import numpy as np
import matplotlib.pyplot as plt

""" Determining range of target function depending on the class probability 
- Any data point that results in the function > 0, it is in class 1
- Otherwise they are classified as 2. 
    E.g. function    =  sin(x1) + sin(x2) + sin(x3) ...   
        data point   = [1,1,1,1,1]        > 0   ==> class 1 
        data point   = [-1,-1,-1,-1,-1]  <= 0   ==> class 2 
"""    
def generate_target_function(num_features, split=0.5):
    if split == 0:
        range = (-np.pi/2, 0)         # class 2 only
    elif split == 1:
        range = (0 , np.pi/2)         # class 1 only 
    else:
        range = (-np.pi/2, np.pi/2)   # 50-50 split between class 1 and class 2
    
    def evaluate(x):
        result = 0
        for i in x:
            result += np.sin(i)        # y = sin(x1) + sin(x2) + .... => ~ num_features * sin(x)
        return result

    return evaluate, range

""" Generate 'n' number of samples with 'f' features in a given range """
def generate_dataset(range, n, f):    
    np.random.seed(124934589)
    min, max = range
    
    # dataset = np.random.uniform(low=min, high=max, size=(n,f))
    # return dataset

    skew = (max-min)/5                  # add misclassifications to dataset
    class_1 = np.random.uniform(low=min+skew, high=max, size=(n, f))
    class_2 = np.random.uniform(low=min, high=max-skew, size=(n, f))
    return class_1, class_2
    

def get_accuracy(dataset, func, predicted_label):
    correct = 0 
    total = 0
    for point in dataset:
        total += 1
        if func(point) > 0 and predicted_label == 1:
            correct += 1
        elif func(point) <= 0 and predicted_label == 2:
            correct += 1 
    return correct/total
        

def get_error(dataset, func, predicted_label):
    wrong = 0 
    total = 0 
    for point in dataset:
        total += 1 
        if func(point) > 0 and predicted_label == 2:
            wrong += 1 
        elif func(point) <= 0 and predicted_label == 1:
            wrong += 1
    return wrong/total 

def get_bayes_error(error, probability):
    min_error = 0
    for err, p in zip(error, probability):
        min_error += err*p 
    return min_error


if __name__=='__main__':

    p = 0.5
    num_f = 5
    num_samples = 100

    func, range = generate_target_function(num_f, p)
    print('If point is greater than this function in all dimensions then it is in class 1, otherwise class 2')
    print(f'function: {func}')

    print('------------------------------')
    print(f'Generating skewed data based on this target function {p} probability between the two classes...')
    # data = generate_dataset(range, num_samples, num_f)
    # print(data.shape)
    # print(data[:5])

    class_1, class_2 = generate_dataset(range, num_samples, num_f)
    print(class_1.shape, class_2.shape)
    print(f'First 5 of data points of class 1: \n{class_1[:5]}')
    acc_1 = get_accuracy(class_1, func, 1)
    print(f'Maximum accuracy using this dataset for class 1: {acc_1}')

    print(f'First 5 of data points of class 2: \n{class_2[:5]}')
    acc_2 = get_accuracy(class_2, func, 2)
    print(f'Maximum accuracy using this dataset for class 2: {acc_2}')
    

    print('------------------------------')
    errors = []
    probabilities = []
    errors.append(get_error(class_1, func, 1))
    print(f'Probability of classifying class 2 as class 1: {errors[0]}')
    probabilities.append(p)
    errors.append(get_error(class_2, func, 2))
    print(f'Probability of classifying class 1 as class 2: {errors[1]}')
    probabilities.append(1-p)
    bayes_error = get_bayes_error(errors, probabilities)
    print(f'Bayes Error of this dataset is: {bayes_error:.2f}')
    




    