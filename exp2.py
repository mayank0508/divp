# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:24:08 2022

@author: Lenovo
"""

# Exp - 02 Fourier Transform of Image 

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('freeza.png',0)

# Finds the 2D Fourier Transform 
f = np.fft.fft2(img)

# Centers the Fourier Spectrum 
# centering the fourier tranform 

fshift = np.fft.fftshift(f)

# Computes the Fourier magnitude in Decibels 

# ABS becauser Fourier Transform is a complex number a+ib 

# ABS = Sqrt(a^2 + b^2) 

magnitude_spectrum = 20*np.log(np.abs(fshift))

# subplot (rows, columns, index)

# Color map will quantize pixel values to its available values 
# Color Quantization 

plt.subplot(121),plt.imshow(img, cmap = 'copper')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()