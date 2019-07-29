import numpy as np
#from labelEncoder import LabelEncoder

# Helper Functions
def prob_di(xi, h):
    return 0


# x:input features (numpy array)
# ph1: probability of h1
# ph2: probability of h2

def target(x, data):
    # calculate p(h|D)  = p(D|h)p(h)/P(D)

    # calculate p(D|h1)
    h1 = 1
    for i in range(x.shape[0]):
        # calculate p(D(i)|h1)
        pdi_h1 = prob_di(x[i], h1)

    return 0

# calculate bayes error rate
