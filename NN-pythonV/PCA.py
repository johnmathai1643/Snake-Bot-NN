import numpy as np
#np.set_printoptions(threshold=np.nan)
import numpy.matlib
from scipy.io import loadmat
from scipy.io import savemat

def PCA(TRAIN_DATA):
	mat = TRAIN_DATA
	mat = 1.0*mat/255;
	u = mat.mean(0)
	u_rep = np.matlib.repmat(u, np.size(mat,0), 1)
	std = np.std(mat, axis=0)
	std_rep = np.matlib.repmat(std, np.size(mat,0), 1)
	mat = mat - u_rep;
	mat = np.divide(mat, std_rep)
	S = ((mat - u_rep).T).dot(mat - u_rep)
	
	#eigenValues,eigenVectors = np.linalg.eig(S)
	#idx = eigenValues.argsort()[::-1]   
	#eig_val = eigenValues[idx]
	#eigenVectors = eigenVectors[:,idx]
	#eig_v= eigenVectors[:,:200];
	eig_v = loadmat('V.mat')
	eig_v = eig_v['V']
	eig_v= eig_v[:,:200]
	#print eig_v
	mat_transform = mat.dot(eig_v);
	
	u_tra = mat_transform.mean(0);
	u_rep = np.matlib.repmat(u_tra, np.size(mat_transform,0), 1)
	std = np.std(mat_transform, axis=0)
	std_rep = np.matlib.repmat(std, np.size(mat_transform,0), 1)
	mat_transform = mat_transform - u_rep;
	mat_transform = np.divide(mat_transform, std_rep)
	savemat('mat_transform1.mat', mdict={'mat_transform1': mat_transform})
	#savemat('eig_v.mat', mdict={'eig_v': eig_v})
	#savemat('eig_val.mat',mdict={'eig_val': eig_val})
	return (mat_transform,eig_v)