import cv2
import numpy as np
import glob
from feature import img_features

images = glob.glob("Images/asphalt_01_25/*.png")

for image in images:
	img = cv2.imread(image)
	features_0, features_90 = img_features(img)
