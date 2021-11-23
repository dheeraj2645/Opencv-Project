# import numpy as np
# import cv2
# image='face.jpg'
# img = cv2.imread(image,1)
# img = cv2.resize(img, (1060, 740))
# cv2.imshow('face',img)
# cv2.waitKey(0)
# converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('face_hsv',converted)
# cv2.waitKey(0)
# fmask = cv2.inRange(converted, np.array([0, 48, 80], dtype = "uint8"),np.array([20, 255, 255], dtype = "uint8"))
# cv2.imshow('face_mask',fmask)
# cv2.waitKey(0)
# contours, hier = cv2.findContours(fmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # create emtpy mask
# mask = np.zeros(fmask.shape[:2], dtype=fmask.dtype)

# # draw all contours larger than 20 on the mask
# mean=0
# for i in contours:
#     mean+=cv2.contourArea(i)
# mean=mean/len(contours)
# largest_areas = sorted(contours, key= cv2.contourArea)
# # mean=np.mean(contours)
# print (mean)
# for c in largest_areas:
#     (x, y, w, h) = cv2.boundingRect(c)
    
#     if (cv2.contourArea(c) >mean):
#         print (cv2.contourArea(c))
#         x, y, w, h = cv2.boundingRect(c)
#         cv2.drawContours(mask, [c], 0, (255), -1)
# # cv2.drawContours(mask, [largest_areas[-1]], 0, (255), -1)
# # apply the mask to the original image
# result = cv2.bitwise_and(fmask,fmask, mask= mask)
# # edges=result.copy()
# #show image
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# # kernel = cv2.getStrum
# skin = cv2.bitwise_and(img, img, mask = result)
# 	# show the skin in the image along with the mask
# cv2.imshow("images", skin)
# cv2.waitKey(0)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# min_YCrCb = np.array([0,133,77],np.uint8)
# max_YCrCb = np.array([235,173,127],np.uint8)
image = cv2.imread("face.jpg")
image = cv2.resize(image, (1060, 740))
# image=cv2.resize(image, None, fx=0.5, fy=0.5)
blur=cv2.medianBlur(image, 19)
# blur = cv2.GaussianBlur(image,(21,21),0)
#print(image)
cv2.imshow('skinYCrcb',blur)
cv2.waitKey(0)
imageYCrCb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
cv2.imshow('skinYCrcb',imageYCrCb)
cv2.waitKey(0)
fmask = cv2.inRange(imageYCrCb,np.array([0,133,77],np.uint8),np.array([235,173,127],np.uint8))
# skinYCrCb = cv2.bitwise_and(image, image, mask = fmask)

# cv2.imshow('image',image)
cv2.imshow('skinYCrcb',fmask)
cv2.waitKey(0)
contours, hier = cv2.findContours(fmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# create emtpy mask
mask = np.zeros(fmask.shape[:2], dtype=fmask.dtype)

# draw all contours larger than 20 on the mask
mean=0
for i in contours:
    mean+=cv2.contourArea(i)
mean=mean/len(contours)
largest_areas = sorted(contours, key= cv2.contourArea)
# mean=np.mean(contours)
print (mean)
for c in largest_areas:
    (x, y, w, h) = cv2.boundingRect(c)
    
    if (cv2.contourArea(c) >mean):
        print (cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(mask, [c], 0, (255), -1)
# cv2.drawContours(mask, [largest_areas[-1]], 0, (255), -1)
# apply the mask to the original image
result = cv2.bitwise_and(fmask,fmask, mask= mask)
# edges=result.copy()
#show image
cv2.imshow("Result", result)
cv2.waitKey(0)
skin = cv2.bitwise_and(image, image, mask = result)
	# show the skin in the image along with the mask
cv2.imshow("images", skin)
cv2.waitKey(0)