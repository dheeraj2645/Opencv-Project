import numpy as np
import cv2
image='Bookshelf.jpg'
img = cv2.imread(image,1)
img = cv2.resize(img, (1060, 740))
cv2.imshow('book',img)
cv2.waitKey(0)
# cv2.imwrite(filename+'_greyscale.png',grey)
for k in range(5,7,2):
    blur = cv2.GaussianBlur(img,(13,13),0)
    # kernel = np.ones((10,10),np.uint8)
    # blur=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #sobel
    cv2.imshow('book',blur)
    cv2.waitKey(0)
    # grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)

    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # cv2.imshow('Sobel Image',grad_x)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Image',grad_y)
    # cv2.waitKey(0)
    # # cv2.imwrite(filename+'_sobel_gradX.png',grad_x)
    # # cv2.imwrite(filename+'_sobel_gradY.png',grad_y)
    # cv2.imshow('Sobel Image',grad)
    # cv2.waitKey(0)
    # cv2.imwrite(filename+'_sobel.png',grad)

    # #canny
    maxi=0
    final=[[]]
    # for i in range(0,100,10):
    #     print (i)
#     for canny in range(30,100,10):
#         edges = cv2.Canny(blur, canny, 2*canny,apertureSize=3)
#         # cv2.imshow('blur_canny', edges)
#         # cv2.waitKey(0),,,,,,,,,,,,,,,,,
#         # edges = cv2.Canny(blur, 20, 20,apertureSize=3)
#         cv2.imshow('canny', edges)
#         cv2.waitKey(0)
#         # contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         # # cv2.imwrite(filename+'_canny.png',edges)
#         # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
#         # cv2.imshow('canny', img)
#         # print (len(contours))

#         # for j in range(10,300,10):
#             # for k in range(10,300,10):
#         # for theta in range(6,360,6):
#         #     print (theta,canny)
#         #     img1=img.copy()
#         #     try:
#         #         lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/theta, threshold=150,lines=np.array([]), minLineLength=100,maxLineGap=1000)
#         #         cv2.waitKey(0)
#         #         a,b,c = lines.shape
#         #         for i in range(a):
#         #             cv2.line(img1, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
#         #         cv2.imshow('houghlines5.jpg',img1)
#         #         cv2.waitKey(0)
#         #     except :
#         #         print (theta)
#         minLineLength = 10
#         maxLineGap = 10
#         for j in range(120,300,10):
            
#             for k in range(10,360,10):
#                 lines = cv2.HoughLines(edges,1,np.pi/k,j,None ,0,0)
#                 img1=img.copy()
#                 print (canny,j,k)
#             # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#                     # print (lines[0])
#                     # try:
#                     #     length=len(lines)
#                     #     print (i,j,k,length)
#                     #     if length>maxi:
#                     #         final[-1]=[i,j,k]
#                     #         print (final[-1])
#                     #         maxi=length
#                 try:      
#                     for line in lines:
                        
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
#                     cv2.imshow('hough_lines',img1)
#                     cv2.imwrite('houghlines3.jpg',img)
#                     cv2.waitKey(0)
#                 except:
#                     print ("")
# # print (maxi,final)
canny=30
j=160
k=180
edges = cv2.Canny(blur, canny, 2*canny,apertureSize=3)
       
cv2.imshow('canny', edges)
cv2.waitKey(0)
lines = cv2.HoughLines(edges,1,np.pi/k,j,None ,0,0)
img1=img.copy()
# print (canny,j,k)


for line in lines:
    
    rho,theta=line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
    # x1,y1,x2,y2=line[0]
    # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('hough_lines',img1)
cv2.imwrite('houghlines3.jpg',img)
cv2.waitKey(0)
print (len(lines)//2) 
cv2.destroyAllWindows()