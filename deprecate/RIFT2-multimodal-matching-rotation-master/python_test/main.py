import cv2
import numpy as np
from featuredetection import feature_detection
from orientation import kpts_orientation  # Comment this out if it does convolution internally

print("Reading images...")

# Read images directly in grayscale
im1 = cv2.imread("ROIs1970_fall_s1_8_p151.png", cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread("ROIs1970_fall_s2_8_p151.png", cv2.IMREAD_GRAYSCALE)

print("Detecting features...")
# Detect features (assuming feature_detection works fine with a single-channel image)
key1, m1, eo1 = feature_detection(im1, s=4, o=6, npt=5000)

# If you want to completely skip orientation (no convolution):
kpts = kpts_orientation(key1,m1,1,96)  # or however you’d like to handle the orientation step

print(kpts)
