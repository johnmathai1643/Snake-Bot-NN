import numpy as np
#np.set_printoptions(threshold=np.nan)
import math
from scipy.io import loadmat

from labelize_transform_Y import labelize_transform_Y
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from nnCostFunction import nnCostFunction
from predict import predict
from PCA import PCA


TRAIN = loadmat('TRAIN.mat')
TEST = loadmat('TEST.mat')
y = loadmat('Y_TRAIN.mat')
y_test = loadmat('Y_TEST.mat')

TRAIN = np.asmatrix(TRAIN['TRAIN'])
TEST = np.asmatrix(TEST['TEST'])
y = np.asmatrix(y['Y_TRAIN'])
y_test = np.asmatrix(y_test['Y_TEST'])

input_layer_size  = 200;
hidden_layer_size = 120;  
num_labels = 40;

transformed_X,eig_v = PCA(TRAIN)
transformed_Y = labelize_transform_Y(transformed_X,num_labels,np.asmatrix(y))

lamb = 0
eta = 0.6;
print("\nInitializing Neural Network Parameters ...\n")

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

for i in range(1,2000):
   J,Theta1,Theta2 = nnCostFunction(num_labels, transformed_X, transformed_Y, lamb, eta, initial_Theta1, initial_Theta2, y)
   initial_Theta1 = Theta1
   initial_Theta2 = Theta2
   print "Iteration number:%s :: Cost Function(J):%s" % (i,J)

pred = predict(Theta1, Theta2, transformed_X)
print pred

TEST = 1.0*TEST/255
mat = TEST
u_test = mat.mean(0)
u_rep_test = np.matlib.repmat(u_test, np.size(mat,0), 1)
std = np.std(mat, axis=0)
std_rep = np.matlib.repmat(std, np.size(mat,0), 1)
mat = mat - u_rep_test;
mat = np.divide(mat, std_rep)
mat_transform = mat.dot(eig_v);
u_tra = mat_transform.mean(0)
u_rep = np.matlib.repmat(u_tra, np.size(mat_transform,0), 1)
std = np.std(mat_transform, axis=0)
std_rep = np.matlib.repmat(std, np.size(mat_transform,0), 1)
mat_transform = mat_transform - u_rep;
mat_transform = np.divide(mat_transform, std_rep)

pred2 = predict(Theta1, Theta2, mat_transform)
#print pred2
print (np.mean(((pred==y)).astype(int))*100)
print (np.mean(((pred2==y_test)).astype(int))*100)