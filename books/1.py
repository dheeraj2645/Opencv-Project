import numpy as np
import cv2
import scipy.ndimage
import scipy.stats
def remove_small_lines(img):
    #find all your connected components (white blobs in your image)
    img=img.astype(np.uint8) 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 5000

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return np.uint8( img2)
def vertical_erode(img, structure_length, iterations, debug=False):
    '''
    Erodes the image with a vertical structure element of length structure_length.
    Used to get rid of lines that are primarily horizontal.
    '''
  
    # img = cv2.imread('j.png',0)
    
    # structure = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) * structure_length
    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    kernel=np.uint8(kernel)
    erosion = cv2.erode(img,kernel,iterations =8)
    dilation=cv2.dilate(erosion,kernel,iterations=10)
    # proc_img = scipy.ndimage.morphology.binary_erosion(
    #     img, structure, iterations)

    # if debug:
    #     print('vertical erode')
    #     # plot_img(proc_img, show=True)
    return dilation
    # return np.uint8(proc_img)
# proc_img
image='Bookshelf1.jpg'
img = cv2.imread(image,1)
img = cv2.resize(img, (1060, 740))
cv2.imshow('book',img)
cv2.waitKey(0)
# cv2.imwrite(filename+'_greyscale.png',grey)
# for k in range(5,50,2):

kernel = np.ones((10,10),np.uint8)
blur=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#sobel
cv2.imshow('book',blur)
cv2.waitKey(0)
grey = cv2.imread(image,0)
grey = cv2.resize(grey, (1060, 740))
gaus_blur = cv2.GaussianBlur(grey,(11,11),0)
# cv2.imshow('book',grey)
# cv2.waitKey(0)
# cv2.imwrite(filename+'_greyscale.png',grey)
# for k in range(5,50,2):
# grey_blur = cv2.GaussianBlur(grey,(11,11),0)
kernel = np.ones((8,8),np.uint8)
grey_blur=cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
#sobel
cv2.imshow('book',grey_blur)
cv2.waitKey(0)
# grey_blur=gaus_blur.copy()
grad_x = cv2.Sobel(grey_blur, cv2.CV_64F, 1, 0, ksize=-1)**2
grad_y = cv2.Sobel(grey_blur, cv2.CV_64F, 0, 1, ksize=-1)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow('Sobel Image',grad_x)
cv2.waitKey(0)
cv2.imshow('Sobel Image',grad_y)
cv2.waitKey(0)
structure_length = 200
iterations = 8
proc_img = vertical_erode(
grad_x, structure_length, iterations)
# cv2.imwrite(filename+'_sobel_gradX.png',grad_x)
# cv2.imwrite(filename+'_sobel_gradY.png',grad_y)
cv2.imshow('Sobel Image',( proc_img))
cv2.waitKey(0)
cv2.imshow('Sobel Image1',remove_small_lines( proc_img))
cv2.waitKey(0)
# cv2.imwrite(filename+'_sobel.png',grad)
blur=grad.copy()
# #canny
maxi=0
final=[[]]
# for i in range(0,100,10):
# # #     print (i)
# for canny in range(100,300,10):
#     edges = cv2.Canny(blur, canny, canny)
# # high_thresh, thresh_im = cv2.threshold(grey_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # lowThresh = 0.3*high_thresh
# # print (lowThresh)
# # edges = cv2.Canny(blur, lowThresh, high_thresh)

#     cv2.imshow('canny', edges)
#     cv2.waitKey(0)
#         # cv2.imshow('blur_canny', edges)
#         # cv2.waitKey(0),,,,,,,,,,,,,,,,,
#     # edges = cv2.Canny(blur, 20, 20,apertureSize=3)
#     # cv2.imshow('canny', edges)
#     # cv2.waitKey(0)
proc_img=np.uint8(proc_img)
contours, hierarchy = cv2.findContours(proc_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     # # cv2.imwrite(filename+'_canny.png',edges)
img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow('con', img)
cv2.waitKey(0)
print (len(contours))

#     # for j in range(10,300,10):
#         # for k in range(10,300,10):
            
#     minLineLength = 10
#     maxLineGap = 10
#     for j in range(100,300,10):
#         lines = cv2.HoughLines(edges,1,np.pi/180,j,None ,0,0)
#         img1=img.copy()
#     # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#             # print (lines[0])
#             # try:
#             #     length=len(lines)
#             #     print (i,j,k,length)
#             #     if length>maxi:
#             #         final[-1]=[i,j,k]
#             #         print (final[-1])
#             #         maxi=length
#         try:      
#             for line in lines:
                
#                 rho,theta=line[0]
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a*rho
#                 y0 = b*rho
#                 x1 = int(x0 + 1000*(-b))
#                 y1 = int(y0 + 1000*(a))
#                 x2 = int(x0 - 1000*(-b))
#                 y2 = int(y0 - 1000*(a))

#                 cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
#                 # x1,y1,x2,y2=line[0]
#                 # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.imshow('hough_lines',img1)
#             cv2.imwrite('houghlines3.jpg',img)
#             cv2.waitKey(0)
#         except:
#             print ("")
# # print (maxi,final)
# cv2.destroyAllWindows()