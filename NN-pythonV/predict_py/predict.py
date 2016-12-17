import numpy as np
from sigmoid import sigmoid
from scipy.io import loadmat

def predict(Theta1, Theta2, X):
	m = np.size(X, 0)
	num_labels = np.size((Theta2, 0))
	p = np.zeros((np.size((X, 1)), 1))

	h1 = sigmoid(np.concatenate((np.ones((m, 1)), X),axis=1).dot(Theta1.T))
	h2 = sigmoid(np.concatenate((np.ones((m, 1)), h1),axis=1).dot(Theta2.T))
	p = np.argmax(h2, axis=1).T
	return p

TEST = loadmat('TEST.mat')
y_test = loadmat('y_test.mat')
Theta1 = loadmat('Theta1.mat')
Theta2 = loadmat('Theta2.mat')
eig_v = loadmat('V.mat')

eig_v = np.asmatrix(eig_v['V'])
TEST = np.asmatrix(TEST['TEST'])
Theta1 = np.asmatrix(Theta1['Theta1'])
Theta2 = np.asmatrix(Theta2['Theta2'])
y_test = np.asmatrix(y_test['y_test'])

eig_v= eig_v[:,:1845]
TEST_TR = TEST.dot(eig_v)

pred = predict(Theta1, Theta2, TEST_TR)
print pred
print (np.mean(((pred==y_test)).astype(int))*100)
