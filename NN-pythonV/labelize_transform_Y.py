import numpy as np
from scipy.io import loadmat

def labelize_transform_Y(transform_X,num_labels,y):
    m = np.size(transform_X, 0)
    Y = np.zeros((m,num_labels))
    for i in range(0,m):
    	Y[i][y[i]] = 1;	
    return Y

#y = loadmat('Y_TRAIN.mat')
#y = np.asmatrix(y)

#print labelize_transform_Y(np.random.randint(255, size=(200, 200)),40,y)

#print labelize_transform_Y(np.random.randint(3, size=(3, 1)),3,np.random.randint(3, size=(3, 1)))