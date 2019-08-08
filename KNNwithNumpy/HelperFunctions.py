import numpy as np

def get_model_results(KNN_model, k_range, weight_func, X, y, print_results=False):
    '''
    Evaluates the KNN model (classifier or regressor) on a range of k values
    using both the euclidean and manhattan distance metrics. Returns two lists.
    The first list contains the evaluation results using the euclidean distance
    metric and the second contains the evaluation results using the manhattan
    distance metric.Optionally, prints the results.
    '''
    if print_results:
        print("{:<9}{:<12}{}".format("k", "Manhattan", "Euclidean"))
        print("------------------------------")
    manhattan_results = []
    euclidean_results = []
    for k in k_range:
        model = KNN_model(n_neighbours=k, dist_metric='manhattan', weight_func=weight_func)
        manhattan_result = model.evaluate(X, y)
        model = KNN_model(n_neighbours=k, dist_metric='euclidean', weight_func=weight_func)
        euclidean_result = model.evaluate(X, y)
        if print_results:
            print("{:<10}{:<12.4f}{:.4f}".format(k, manhattan_result, euclidean_result))
        manhattan_results.append(manhattan_result)
        euclidean_results.append(euclidean_result)
    return manhattan_results, euclidean_results


def PCA(X, n_components, print_variance=True):
    """
    Performs principal component analysis to reduce the number of
    features in X. We will use this just for visualization purposes so we
    don't need to implement the inverse transform to get the the approximate
    original X back from the reduced X.
    """
    # Singular Value Decomposition
    cov = np.cov(np.transpose(X))
    U, s, V = np.linalg.svd(cov) # U = eigenvectors, s = eigenvalues
    # Calcuate variance retained
    if print_variance:
        num = 0
        den = np.sum(s)
        for i in range(n_components):
            num += s[i]
        print("Variance retained: {:.2f}".format(num / den))
    # Reduce the number of dimensions
    X_reduced = np.matmul(X, U[:, :n_components])
    return X_reduced

if __name__ == "__main__":
    pass
