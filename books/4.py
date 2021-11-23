import numpy as np
import cv2
def auto_canny(image, sigma=0.33):
     # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
img = cv2.imread('Bookshelf.jpg',1)
img=cv2.resize(img,(1080,760))
gray = cv2.imread('Bookshelf.jpg',0)
gray=cv2.resize(gray,(1080,760))
kernel = np.ones((10,10),np.uint8)
gray=cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
# high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# lowThresh = 0.5*high_thresh
# print (lowThresh)
# edges = cv2.Canny(gray, lowThresh, high_thresh)
# edges=auto_canny(gray)
cv2.imshow('edges-50-150.jpg',edges)
minLineLength=200
for theta in range(6,360,6):
    print (theta)
    img1=img.copy()
    try:
        lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/theta, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=300)
        cv2.waitKey(0)
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(img1, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('houghlines5.jpg',img1)
        cv2.waitKey(0)
    except :
        print (theta)