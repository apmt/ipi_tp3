import cv2
import numpy as np
import glob
from feature import img_features

# 6 features = 3 de Y_Luminance(gray) + 3 de CR_chrominance

images = glob.glob("../Images/asphalt_01_25/*.png")

ASFALTO, DANGER, GRASS = 0, 1, 2
zonas = ASFALTO, DANGER, GRASS

Quantidade_imagens = len(images)
Quantidade_features = 6
matriz_de_features = np.zeros((len(zonas), Quantidade_features, Quantidade_imagens))
Centros_de_massa = np.zeros((len(zonas), Quantidade_features))

# ASFALTO
images = glob.glob("../Images/asphalt_01_25/*.png")
for index, image in enumerate(images, start=0):
	img = cv2.imread(image)
	features = img_features(img)
	matriz_de_features[ASFALTO, :, index] = features
Centros_de_massa[ASFALTO, :] = np.mean(matriz_de_features[ASFALTO], axis=1)

print 'Carregando extraction 30%'
# DANGER
images = glob.glob("../Images/danger_01_25/*.png")
for index, image in enumerate(images, start=0):
	img = cv2.imread(image)
	features = img_features(img)
	matriz_de_features[DANGER, :, index] = features
Centros_de_massa[DANGER, :] = np.mean(matriz_de_features[DANGER], axis=1)

print 'Carregando extraction 60%'
# GRASS
images = glob.glob("../Images/grass_01_25/*.png")
for index, image in enumerate(images, start=0):
	img = cv2.imread(image)
	features = img_features(img)
	matriz_de_features[GRASS, :, index] = features
Centros_de_massa[GRASS, :] = np.mean(matriz_de_features[GRASS], axis=1)

print 'Carregando extraction 90%'
print Centros_de_massa
np.save("Centros_de_massa", Centros_de_massa)
