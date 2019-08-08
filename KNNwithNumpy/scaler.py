import pandas as pd
from scipy.io import arff

class MinMaxScaler:
    def __init__(self, feature_range=(0,1)):
        min, max = feature_range
        self.feature_min = min
        self.feature_max = max

    def fit_transform(self, X):
        X = X.to_numpy().astype(np.float64)
        min = X.min(axis=0)
        max = X.max(axis=0)
        X_std = (X - min) / (max - min)
        X_scaled = X_std * (self.feature_max - self.feature_min) + self.feature_min
        return X_scaled


def test_MinMaxScaler():
    # Load data from autos.aff
    data_set = arff.loadarff('autos.arff')
    data = pd.DataFrame(data_set[0])

    # Remove the old categorical labels
    data.drop(['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders',
                'fuel-system'], axis=1, inplace=True)

    # Remove missing values
    filtered = data.dropna()
    print(filtered.shape)

    # Separate labels & x-data
    labels = filtered['price'].to_numpy()
    x_data = filtered.drop('price', axis=1)
    print(x_data[:5])

    # Transform data
    scaler = MinMaxScaler()
    scaled_x_data = scaler.fit_transform(x_data)
    print(scaled_x_data[:5])

if __name__=='__main__':
    test_MinMaxScaler()
