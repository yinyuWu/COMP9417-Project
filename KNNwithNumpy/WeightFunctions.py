import numpy as np

def uniform(distances):
    '''
    weight = 1
    Example:
    distances = [1, 2, 3, 4]
    weights   = [1, 1, 1, 1]
    '''
    return(np.ones(distances.shape))

def inverse_squared_dist(distances):
    '''
    weight = 1 / (distance + 0.0001)
    Example:
    distances = [1, 2, 3, 4]
    weights   = [1, 0.25, 0.11, 0.06]
    '''
    return 1 / (np.square(distances) + 0.0001)

def inverse_dist(distances):
    '''
    weight = 1 / (distance**2 + 0.0001)
    Example:
    distances = [1, 2, 3, 4]
    weights   = [1, 0.5, 0.33, 0.25]
    '''
    return 1 / (distances + 0.0001)

def exponential(distances):
    '''
    weight = exp(-distance)
    Example:
    distances = [1, 2, 3, 4]
    weights = [0.37, 0.14, 0.05, 0.02]
    NOTE: For large distances the weight will be too small to store as a float.
    '''
    return np.exp(-distances)

def dudani(distances):
    '''
    weight = k > 1: (max(distances) - distance) / (max(distances) - min(distances) + 0.0001)
            k == 1: 1
    Example:
    distances = [1, 2, 3, 4]
    weights = [1, 0.67, 0.33, 0]
    '''
    if distances.shape[0]==1:
        return np.array([1])
    else:
        return (distances[-1]-distances) / (distances[-1]-distances[0] + 0.0001) # +0.0001 in case all k nearest have same dist

if __name__ == "__main__":
    pass
