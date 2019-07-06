import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from distance import Manhattan, Euclidean

# knn for numeric prediction

class KNN_Numeric:
    def __init__(self, x_data, labels, k_neighbours):
        self.x_data = x_data
        self.labels = labels
        self.k_neighbours = k_neighbours
    
    def predict(self, ux):
        # Find nearest k neighbours by Euclidean/Manhattan distance
        n = self.x_data.shape[0]    
        dists = np.zeros((n, 2))
        for i in range(n):                                              
            dists[i][0] = Euclidean(self.x_data[i], ux)
            dists[i][1] = self.labels[i]

        sorted_dists = dists[np.argsort(dists[:,0])]      # Sort by distances which is 1st column              
        neighbours = sorted_dists[:self.k_neighbours]     # Take only the closest k neighbours
        return np.mean(neighbours[:,1])                   # Return mean of 2nd column which is labels of neighbours

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
    

def main():
    # Load data from autos.aff
    data_set = arff.loadarff('autos.arff')
    data = pd.DataFrame(data_set[0])
    
    # Remove any categorical labels (Note: temporary as we can encode these categorical labels later)
    data.drop(['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 
                'fuel-system'], axis=1, inplace=True)

    # Remove missing values
    filtered = data.dropna()
    print(filtered.shape)

    # Separate labels & x-data
    labels = filtered['price']
    x_data = filtered.drop('price', axis=1)

    # TEMPORARY TEST FOR KNN NUMERIC
    Test_KNN_Numeric(x_data, labels.to_numpy())

    #Manhattan_Test(data)
    #Euclidean_Test(data)

if __name__ == "__main__":
    main()