import random
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

	
img1 = cv.imread("../images_count/5.jpg")
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
step = 5
ptr = 0
start = 0
start1 = 0
end = (edges.shape[1]/step) - 1
end1 = (edges.shape[1]/step) - 1

count = 0
while ptr < step-1:
	j = 0
	while j < row-1:
	 	sec = edges[j:j+2,start:end+1]
	 	sec = sec/255
	 	
		for i in range(start1, end1+1):
			if(any(sec[:,i])):
				sec[:,i] = np.ones(sec.shape[0])
		edges[j:j+2,start:end+1] = sec
		j = j+2
#	cv.imshow("image", edges)
#	cv.waitKey(0)
	for i in range(0 , row):
		#print edges[i]
		sval = sum(edges[i,start:end+1])
		if(sval < (end-start+1)/4):
			edges[i, start:end+1] = 0
		else:
			edges[i, start:end+1] = 255
	count1 = 0
	for i in range(0, row-1):
		if edges[i,start] != edges[i+1,start]:
			count1+=1
			count += 1

	#print count
	#print math.ceil(count1/2.0)
	ptr += 1
	start += edges.shape[1]/step
	end += edges.shape[1]/step

#start += edges.shape[1]/step
end = edges.shape[1] - 1
j = 0
while j < row-1:
 	sec = edges[j:j+2,start:end + 1]
 	sec = sec/255
 	
	for i in range(start1, sec.shape[1]):
		if(any(sec[:,i])):
			sec[:,i] = np.ones(sec.shape[0])
	edges[j:j+2,start:end + 1] = sec
	j = j+2

#print start ," ", end
#cv.imshow("image", edges)
#cv.waitKey(0)

for i in range(0 , row):
	#print edges[i]
	sval = sum(edges[i,start:end + 1])
	if(sval < (end-start + 1)/4):
		edges[i, start:end + 1] = 0
	else:
		edges[i, start:end + 1] = 255
count1 = 0
for i in range(0, row-1):
	if edges[i,start] != edges[i+1,start]:
		count1 +=1
		count += 1

#print count
#print math.ceil(count1/(2.0))
print int(round(count/(2.0*step)))


plt.subplot(141),plt.imshow(img2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(142),plt.imshow(blur_img,cmap = 'gray')
plt.title('Gaussian Filtering'), plt.xticks([]), plt.yticks([])


plt.subplot(143),plt.imshow(edges1,cmap = 'gray')
plt.title('Otsu Thresholding'), plt.xticks([]), plt.yticks([])

plt.subplot(144),plt.imshow(edges,cmap = 'gray')
plt.title('Counting Sarees'), plt.xticks([]), plt.yticks([])

plt.show()
cv.waitKey(0)
