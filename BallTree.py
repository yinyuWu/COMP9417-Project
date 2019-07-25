import numpy as np
from distance import Euclidean
from sklearn.preprocessing import LabelEncoder

'''
function construct_balltree is
       input: 
           D, an array of data points
       output: 
           B, the root of a constructed ball tree
       if a single point remains then
           create a leaf B containing the single point in D
           return B
       else
           let c be the dimension of greatest spread
           let p be the central point selected considering c
           let L,R be the sets of points lying to the left and right of the median along dimension c
           create B with two children: 
               B.pivot = p
               B.child1 = construct_balltree(L),
               B.child2 = construct_balltree(R),
               let B.radius be maximum distance from p among children
           return B
       end if
   end function
'''
'''
data = x_data + label
centroid = x_data
'''

class BallTree():
    def __init__(self, data):
        if(data.shape[0] <= 1):
            self.data = data
            self.centroid = data[:-1]
            self.radius = 0
            self.left_child = None
            self.right_child = None
        else:
            self.data = data
            dim_c = self.find_largest_dim(data[:-1])
            sorted_data = self.find_sorted_data(data, dim_c)
            size = data.shape[0]
            self.centroid = sorted_data[int(size/2)][:-1]
            self.left_child = BallTree(np.asarray(sorted_data[:int(size/2)]))
            self.right_child = BallTree(np.asarray(sorted_data[int(size/2)+1:]))
            self.radius = self.find_radius()
    
    def find_largest_dim(self, data):
        return np.argmax(data.max(0)- data.min(0))
    
    def find_sorted_data(self, data, dim):
        return sorted(data, key=lambda x : x[dim])
    
    def find_radius(self):
        radius = 0
        for each in self.data:
            if (Euclidean(self.centroid, each[:-1]) > radius):
                radius = Euclidean(self.centroid, each[:-1])
        return radius
    
    def getData(self):
        return self.data
    
    def printInfo(self):
        print("Left Children: ")
        print(self.left_child.getData())
        print("################")
        print("Right Children: ")
        print(self.right_child.getData())
        print("################")
        print("Central point: " + str(self.centroid))
        print("################")
        print("Radius: " + str(self.radius))


def test():
    label = np.asarray(['a', 'b', 'a', 'a'])
    data1 = np.asarray([[1.1,2.5,3.0,4.7],[5.1,7.2,1.3,2.1],[4.5,3.6,2.2,1.2],[1.1,4.9,8.6,4.7]])
    le = LabelEncoder()
    label = le.fit_transform(label)
    #print(label)
    data = np.concatenate([data1, np.ones((data1.shape[0],1),dtype=data1.dtype)], axis=1)
    for i in range(len(label)):
        data[i, -1] = label[i]
    print(data)
    ballTree1 = BallTree(data)
    ballTree1.printInfo()

if __name__ == "__main__":
    test()
