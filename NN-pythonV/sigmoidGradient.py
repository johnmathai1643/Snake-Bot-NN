import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
	# g = np.zeros((np.size(z,0),np.zeros(z,1))
	g = np.multiply(sigmoid(z),(1.0-sigmoid(z)))
	return g