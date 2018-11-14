import numpy as np
from luminance import Yluminance
import math

img = cv2.imread('final.bmp')
def img_features (img):
	h, w, bpp = np.shape(img)
	img_gray = np.zeros((h, w), dtype=int)
	GLCM_0 = np.zeros((256, 256), dtype=float)
	GLCM_90 = np.zeros((256, 256), dtype=float)
	for y in range (0, h):
		for x in range (0, w):
			img_gray[y][x] = Yluminance(img[y][x])
			if (x>0):
				GLCM_0[img_gray[y][x-1]][img_gray[y][x]] += 1.0/float(h*(w-1))
	Energy_0 = np.sum(GLCM_0*GLCM_0)
	Contrast = 0.0
	Homogeneity = 0.0
	Entropy = 0.0
	for i in range (0, 256):
		for j in range (0, 256):
			Contrast += GLCM_0[i][j] * (i-j) * (i-j)
			Homogeneity += GLCM_0[i][j]/(1+((i-j)*(i-j)))
			Entropy -= GLCM_0[i][j]*math.log10(GLCM_0[i][j])
	return Energy_0, Contrast, Homogeneity, Entropy