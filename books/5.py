import numpy as np
import cv2

image = cv2.imread('Bookshelf.jpg',0)
image=cv2.resize(image,(1080,760))
kernel = np.ones((8,8),np.uint8)
image=cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(image,50,150,apertureSize = 3)
cv2.imshow('edges-50-150.jpg',edges)
cv2.waitKey(0)
# find contours
contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# create emtpy mask
mask = np.zeros(image.shape[:2], dtype=image.dtype)

# draw all contours larger than 20 on the mask

largest_areas = sorted(contours, key= cv2.contourArea)
for c in largest_areas[len(largest_areas)//2+100:]:
    # print (c)
    (x, y, w, h) = cv2.boundingRect(c)
    if h > 100 or w>200 :
    
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(mask, [c], 0, (255), -1)
# apply the mask to the original image
result = cv2.bitwise_and(image,image, mask= mask)
edges=result.copy()
#show image
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.imshow("Image", edges)

cv2.waitKey(0)
minLineLength=200
for theta in range(6,360,6):
    print (theta)
    img1=image.copy()
    try:
        lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/theta, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=300)
        cv2.waitKey(0)
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(img1, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('houghlines5.jpg',img1)
        cv2.waitKey(0)
    except :
        print (theta)
cv2.destroyAllWindows() 