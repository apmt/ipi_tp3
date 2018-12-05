import cv2
import numpy as np
import glob
from feature_S import img_features

legenda = ("Energy_GRAY", "Homogeneity_GRAY", \
"Entropy_GRAY", "Constrast_GRAY" \
"Energy_CR", "Homogeneity_CR", \
"Entropy_CR", "Constrast_CR")
# 8 features = 4 de Y_Luminance + 4 de CR_chrominance

images = glob.glob("../../Images/asphalt_01_25/*.png")

Quantidade_imagens = 3*len(images)
Quantidade_features = 8
matriz_de_features = np.zeros((Quantidade_features, Quantidade_imagens))

# ASFALTO
images = glob.glob("../../Images/asphalt_01_25/*.png")
for index, image in enumerate(images, start=0):
	img = cv2.imread(image)
	features = img_features(img)
	matriz_de_features[:, index] = features

print 'Carregando 30%'
# DANGER
images = glob.glob("../../Images/danger_01_25/*.png")
for index, image in enumerate(images, start=25):
	img = cv2.imread(image)
	features = img_features(img)
	matriz_de_features[:, index] = features

print 'Carregando 60%'
# GRASS
images = glob.glob("../../Images/grass_01_25/*.png")
for index, image in enumerate(images, start=50):
	img = cv2.imread(image)
	features = img_features(img)
	matriz_de_features[:, index] = features

print 'Carregando 90%'
Correlation = abs(np.corrcoef(matriz_de_features))
print Correlation
Correlation[(range(Quantidade_features)),(range(Quantidade_features))] = 0
index_feat_ingnoradas = np.unravel_index(np.argmax(Correlation), Correlation.shape)
print "Ignorar uma das features por alta Correlation:"
print legenda[index_feat_ingnoradas[0]]
print legenda[index_feat_ingnoradas[1]]

Correlation[index_feat_ingnoradas] = 0
index_feat_ingnoradas = np.unravel_index(np.argmax(Correlation), Correlation.shape)
Correlation[index_feat_ingnoradas] = 0

index_feat_ingnoradas = np.unravel_index(np.argmax(Correlation), Correlation.shape)
print "Ignorar uma das features por alta Correlation:"
print legenda[index_feat_ingnoradas[0]]
print legenda[index_feat_ingnoradas[1]]
