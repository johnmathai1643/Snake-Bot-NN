import numpy as np
import math
from scipy.io import loadmat

from labelize_transform_Y import labelize_transform_Y
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from predict import predict

def nnCostFunction(num_labels,X,Y,lamb,eta,Theta1,Theta2,y):
	m = np.size(X, 0)
	Theta1_grad = np.zeros((np.size(Theta1,0),np.size(Theta1,1)))
	Theta2_grad = np.zeros((np.size(Theta2,0),np.size(Theta2,1)))
	a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
	z2 = a1.dot(Theta1.T)
	a2 = sigmoid(z2)
	a2 = np.concatenate((np.ones((np.size(a2,0), 1)), a2),axis=1)
	z3 = a2.dot(Theta2.T)
	a3 = sigmoid(z3)
	hThetaX = a3
	
	J = 1.0/m*np.sum(np.multiply(np.multiply(-Y,np.log(hThetaX))-(1-Y),np.log(1-hThetaX)))

	for t in range(m):
		a1 = np.concatenate((np.asmatrix(1), X[t,].T),axis=0)
 		z2 = Theta1.dot(a1)
 		a2 = sigmoid(z2)
 		a2 = np.concatenate((np.asmatrix(1), a2),axis=0)
 		z3 = Theta2.dot(a2)
 		a3 = sigmoid(z3)

		Y = (np.arange(num_labels)==y[t]).astype(int).T
 		delta3 = a3 - Y*1.0
        delta2 = np.multiply((Theta2.T).dot(delta3),np.concatenate((np.asmatrix(1),sigmoidGradient(z2)), axis=0))
        delta2 = delta2[1:]
        Theta1_grad =  Theta1_grad + delta2.dot(a1.T)
        Theta2_grad =  Theta2_grad + delta3.dot(a2.T)
	
	Theta1_grad = Theta1_grad/m
	Theta2_grad = Theta2_grad/m
	
	Theta1 = Theta1 - eta*Theta1_grad
	Theta2 = Theta2 - eta*Theta2_grad

	return (J,Theta1,Theta2)