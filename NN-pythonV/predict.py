import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
	m = np.size(X, 0)
	num_labels = np.size((Theta2, 0))
	p = np.zeros((np.size((X, 1)), 1))

	h1 = sigmoid(np.concatenate((np.ones((m, 1)), X),axis=1).dot(Theta1.T))
	h2 = sigmoid(np.concatenate((np.ones((m, 1)), h1),axis=1).dot(Theta2.T))
	p = np.argmax(h2, axis=1).T
	#p = (np.amax(h2, axis=1)).T
	return p
