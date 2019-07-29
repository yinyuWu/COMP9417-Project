import numpy as np
import pandas as pd
from scipy.io import arff


# Helper Functions
def cnt_h(h, label):
    cnt = 0
    for each in label:
        if each == h:
            cnt += 1
    return cnt

# xi: i feature of unknown x
def prob_di(i, xi, x_data, label, h):
    cnt = 0
    total = cnt_h(h, label)
    for j in range(len(label)):
        if (label[j] == h and x_data[j][i] == xi):
            cnt+=1
    p = cnt/total
    prob = (cnt + p)/(1+total)
    return prob


# x:input features (numpy array)
# ph1: probability of h1
# ph2: probability of h2

def target(x, x_data, label):
    # calculate p(h|D)  = p(D|h)p(h)/P(D)

    # calculate p(D|h1)
    h1 = label[0]
    prob1 = 1
    for i in range(x.shape[0]):
        # calculate p(D(i)|h1) = (nc + mp)/(n+m), m: weight, n: #train that h=hi, nc: #train that h=hi&x[i]=xi
        # p: pior estimate for p(D(i)|h), assume weight = 1
        pdi_h1 = prob_di(i, x[i], x_data, label, h1)
        prob1 = prob1 * pdi_h1
    
    # calculate p(D|h2)
    h2 = label[1]
    prob2 = 1
    for k in range(x.shape[0]):
        # calculate p(D(i)|h1)
        pdi_h2 = prob_di(k, x[k], x_data, label, h2)
        prob2 = prob1 * pdi_h2
    
    return h1 if prob1>=prob2 else h2

def test():
    data_set = arff.loadarff('ionosphere.arff')
    data = pd.DataFrame(data_set[0]).to_numpy()
    x_data = data[:, :-1]
    labels = data[:, -1]
    test_size = int(len(labels)*0.25)
    print("test size: "+str(test_size))
    x_train = x_data[test_size:]
    x_test = x_data[:test_size]
    y_train = labels[test_size:]
    y_test = labels[:test_size]
    print("train size: " + str(x_train.shape[0]))
    cor = 0
    for i in range(x_test.shape[0]):
        pre = target(x_test[i], x_train, y_train)
        if (pre == y_test[i]):
            cor += 1
    print("Accuracy of target function is: " + str(cor/test_size))

def prob_x(xi, x_train):
    cnt = 0
    for each in x_train:
        if xi==each:
            cnt+=1
    return cnt/x_train.shape[0]

def prob_error(x, predict, x_train, labels):
    cnt = 0
    total = 0
    for i in range(x_train.shape[0]):
        if (x_train[i] == x):
            total += 1
        if (x_train[i] == x and labels[i] != predict):
            cnt += 1
    return cnt/total

def bayes_error(x, x_train, labels, predict):
    # sum(p(error|x)p(x)) = p(error)
    # calculate p(x)
    px = prob_x(x, x_train)
    p_error_x = prob_error(x, predict, x_train, labels)
    return px*p_error_x

if __name__ == "__main__":
    test()


