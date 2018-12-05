import numpy as np
from luminance import Yluminance
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
	KB, KG, KR = 114, 587, 299
	B, G, R = (0, KB), (1, KG), (2, KR)
	cores = B, G, R
	ind, K = 0, 1
	normalizar_0 = h*(w-1)
	normalizar_90 = w*(h-1)
	for cor in cores:
		img_gray += (img[:, :, cor[ind]]*cor[K])/1000
	GLCM_0 = np.zeros((256, 256), dtype=float)
	GLCM_90 = np.zeros((256, 256), dtype=float)
	np.add.at(GLCM_0, (img_gray[:, 0:(w-1)], img_gray[:, 1:w]), 1.0)
	np.add.at(GLCM_90, (img_gray[1:h, :], img_gray[0:(h-1), :]), 1.0)
	GLCM_0 /= normalizar_0
	GLCM_90 /= normalizar_90
	print 'Energy 90', Energy(GLCM_90)
	print 'homo 0', Homogeneity(GLCM_0)
	print 'homo 90', Homogeneity(GLCM_90)
	print 'Entropy 0', Entropy(GLCM_0)
	print 'Entropy 90', Entropy(GLCM_90)
	print 'Constrast 0', Constrast(GLCM_0)
	print 'Constrast 90', Constrast(GLCM_90)
	return (Energy(GLCM_0), Homogeneity(GLCM_0),\
	Entropy(GLCM_0), Constrast(GLCM_0)),\
	(Energy(GLCM_90), Homogeneity(GLCM_90),\
	Entropy(GLCM_90), Constrast(GLCM_90));
