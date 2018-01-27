import random
import cv2 as cv
import argparse
import glob

import numpy as np
import math
from matplotlib import pyplot as plt

#img = cv.imread("5_3.jpg",0)
img1 = cv.imread("../images_count/1.jpg")
img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

kernel = np.ones((7,7), np.uint8)
blur_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    
edges = cv.Canny(blur_img,100,200)
edges1 = cv.Canny(blur_img,100,200)

#edges = [ sum ]
row, col = edges.shape

j = 0

print row, " " , col


while j < row-1:
 	sec = edges[j:j+2]
 	sec = sec/255
 	count = 0
	for i in range(0, sec.shape[1]):
		if(any(sec[:,i])):
			sec[:,i] = np.ones(sec.shape[0])

	edges[j:j+2] = sec
	j = j+2

for i in range(0 , row):
	#print edges[i]
	sval = sum(edges[i])
	if(sval < col/4):
		edges[i] = 0
	else:
		edges[i] = 255	 

count = 0
for i in range(0, row-1):
	if edges[i,0] != edges[i+1,0]:
		count += 1		

#print count
print math.ceil(count/2.0)

plt.subplot(131),plt.imshow(img2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges1,cmap = 'gray')
plt.title('Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges,cmap = 'gray')
plt.title('Count of sarees'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
