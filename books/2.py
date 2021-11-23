import numpy as np
import cv2
image='Bookshelf.jpg'
img = cv2.imread(image,0)
color=cv2.imread(image,1)
img = cv2.resize(img, (1060, 740))
color = cv2.resize(color, (1060, 740))
cv2.imshow('book',img)
cv2.waitKey(0)
# cv2.imwrite(filename+'_greyscale.png',grey)
# for i in range(1,50,2):
        # print (i)
# blur = cv2.GaussianBlur(img,(20,20),0)
kernel = np.ones((10,10),np.uint8)
blur=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#sobel
cv2.imshow('book',blur)
cv2.waitKey(0)
# for i in range(20,100,10):
high_thresh, thresh_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*high_thresh
print (lowThresh)
edges = cv2.Canny(blur, lowThresh, high_thresh)

cv2.imshow('canny', edges)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cv2.imwrite(filename+'_canny.png',edges)
img1 = cv2.drawContours(color.copy(), contours, -1, (0,255,0), 3)
cv2.imshow('canny1', img1)
cv2.waitKey(0)
print (len(contours))
        # for j in range(140,300,10):
        #         lines = cv2.HoughLines(edges,1,np.pi/180,j,None ,0,0)
        #         img1=img.copy()
        # # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        #         # print (lines[0])
        #         # try:
        #         #     length=len(lines)
        #         #     print (i,j,k,length)
        #         #     if length>maxi:
        #         #         final[-1]=[i,j,k]
        #         #         print (final[-1])
        #         #         maxi=length
        #         try:      
        #                 for line in lines:
                                
        #                         rho,theta=line[0]
        #                         a = np.cos(theta)
        #                         b = np.sin(theta)
        #                         x0 = a*rho
        #                         y0 = b*rho
        #                         x1 = int(x0 + 1000*(-b))
        #                         y1 = int(y0 + 1000*(a))
        #                         x2 = int(x0 - 1000*(-b))
        #                         y2 = int(y0 - 1000*(a))

        #                         cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
        #                         # x1,y1,x2,y2=line[0]
        #                         # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        #                 cv2.imshow('hough_lines',img1)
        #                 cv2.imwrite('houghlines3.jpg',img)
        #                 cv2.waitKey(0)
        #         except:
        #                 print ("")