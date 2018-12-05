import numpy as np
from math import log

def Energy(GLCM):
	energy = np.sum(GLCM**2)
	return energy
def Constrast(GLCM):
	ind = np.indices(GLCM.shape)
	i = ind[0]
	j = ind[1]
	constrast = np.sum(GLCM*((i-j)**2))
	return constrast
def Homogeneity(GLCM):
	ind = np.indices(GLCM.shape)
	i = ind[0]
	j = ind[1]
	homogeneity = np.sum(GLCM/(1+abs(i-j)))
	return homogeneity
def Entropy(GLCM):
	entropy = np.sum(GLCM*(-np.log(np.where(GLCM[:]!=0, GLCM, 1))))
	return entropy
def img_features (img_aux):
	img = img_aux.copy()
	h, w, aux = img.shape
	img_gray = np.zeros((h, w), dtype=int)
	img_CR = np.zeros((h, w), dtype=int)
	KB, KG, KR = 114, 587, 299
	CR_KB, CR_KG, CR_KR = -81, -418, 500 
	B, G, R = (0, KB, CR_KB), (1, KG, CR_KG), (2, KR, CR_KR)
	cores = B, G, R
	ind, Y_K, CR_K= 0, 1, 2
	normalizar = h*(w-1)
	for cor in cores:
		img_gray += (img[:, :, cor[ind]]*cor[Y_K])/1000
		img_CR += (img[:, :, cor[ind]]*cor[CR_K])/1000
	img_CR += 128
	GLCM_Gray = np.zeros((256, 256), dtype=float)
	np.add.at(GLCM_Gray, (img_gray[:, 0:(w-1)], img_gray[:, 1:w]), 1.0)
	GLCM_Gray /= normalizar

	GLCM_CR = np.zeros((256, 256), dtype=float)
	np.add.at(GLCM_CR, (img_CR[:, 0:(w-1)], img_CR[:, 1:w]), 1.0)
	GLCM_CR /= normalizar

	return Energy(GLCM_Gray), Entropy(GLCM_Gray), Constrast(GLCM_Gray), \
	Energy(GLCM_CR), Entropy(GLCM_CR), Constrast(GLCM_CR);
