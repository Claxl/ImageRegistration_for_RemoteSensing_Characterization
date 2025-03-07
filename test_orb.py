import cv2
import numpy as np
import matplotlib.pyplot as plt

orb = cv2.ORB_create()

im1 = cv2.imread('DATASET/RemoteSensing/SAR_Optical/SO1b.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('DATASET/RemoteSensing/SAR_Optical/SO1b.png', cv2.IMREAD_GRAYSCALE)

kp1, des1 = orb.detectAndCompute(im1, None)
kp2, des2 = orb.detectAndCompute(im2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('out.png', img3)    