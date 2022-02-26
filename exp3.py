# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:35:53 2022

@author: Lenovo
"""

# Image Smoothing and Sharpening 

# 
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('ney.jfif')

# Rearranges the color channels from B,G, R back to R, G, B 

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Kernel of ones of size 5x5 - Uniform Smoothing 

kernel = np.ones((5,5),np.float32)/25

# -1 will give the output image depth as same as the input image
# 3 channel input will give 3 channel output  - color - color 
# 1 channel input will give 1 channel output - color - gray 

dst = cv2.filter2D(img,-1,kernel)

plt.figure
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Smoothing')
plt.xticks([]), plt.yticks([])
plt.show()

# Gaussian Blur Smoothing 

# 0 = sigma or variance of Gaussian Kernel 

blur = cv2.GaussianBlur(img,(5,5),0)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Gaussian Smoothing')
plt.xticks([]), plt.yticks([])
plt.show()

#sharpening mask 3 x 3  - Laplacian Sharpening 
# Laplacian Mask 

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

dst = cv2.filter2D(img, -1, kernel)

plt.figure
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Sharpenend')
plt.xticks([]), plt.yticks([])
plt.show()