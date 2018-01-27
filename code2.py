import random
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
	

	
img1 = cv.imread("../images_count/8.jpg")
img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)	


kernel = np.ones((7,7), np.uint8)
#blur_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # Open (erode, then dilate)
blur_img = cv.GaussianBlur(img,(3,3),0)    

#otsu thresholding
ret2,edges = cv.threshold(blur_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

ret2,edges1 = cv.threshold(blur_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


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

if(edges[0,0] == edges[1,0] == 255 or edges[row-2,0] == edges[row-1,0] == 255):
	count += 1
#print count
print int(round(count/2.0))



plt.subplot(221),plt.imshow(img2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(blur_img,cmap = 'gray')
plt.title('Gaussion Blurring'), plt.xticks([]), plt.yticks([])


plt.subplot(223),plt.imshow(edges1,cmap = 'gray')
plt.title('Otsu Thresholding'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(edges,cmap = 'gray')
plt.title('Counting Sarees'), plt.xticks([]), plt.yticks([])

plt.show()
cv.waitKey(0)
