## COMP9417-Project
UNSW COMP9417 project: Topic2.2 Nearest Neighbour(kNN).

# instructions:

* KNN is an abstract parent class of both KNN classification and KNN numeric prediction.
* labelEncoder.py is our own class to convert categorical labels into integers.
* scaler.py is our own MinMaxScaler
* KNNwithNumpy works alone and contains all necessary functions and classes for naive KNN.


1. Classification: KNN_Class.py contains both naive KNN algorithm, weighted KNN algorithm for classification and leave-one-out cross validation of KNN classifiers.

2. Numeric Prediction: KNN_Numeric.py contains both naive KNN algorithm and weighted KNN algorithm for numeric prediction and leave-one-out cross validation.

3. Target functions and generated data: Target.py contains target functions and functions related to generated data.

4. KNNwithNumpy/Testing.ipynb contains most of the testing we did. notebook.ipynb contains the remaining tests.

5. BallTree.py is a data structure of ball tree which is used to find k nearest neighbours in KNN classification.

# requirements:
numpy,
scipy,
pandas,
sklearn,
jupyter.
