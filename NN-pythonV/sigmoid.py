import numpy as np

def sigmoid(z):
	z = np.asmatrix(z)
	g = np.divide(np.ones((np.size(z,0),np.size(z,1))).astype(float),(1.0 + np.exp(-z)))
	return g