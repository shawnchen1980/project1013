# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:30:52 2019

@author: shawn
"""
import numpy as np
from sklearn.metrics import pairwise
import cv2  
#读取图像文件数据  
img = cv2.imread('output.jpg')  
#生成灰度图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#灰度翻转，这里显示出numpy数组的优势，翻转操作非常容易
gray=255-gray
#阈值化操作，图像二元化,如果需要黑白颠倒，看上一句
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
#找包围线，包围线是绕着白色区域的，可能会返回多条包围线
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#找到包围面积最大的包围线
segmented = max(contours, key=cv2.contourArea)
#根据包围线获取凸围线
hull=[cv2.convexHull(segmented,False)]
  
chull = cv2.convexHull(segmented)

# find the most extreme points in the convex hull
extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

# find the center of the palm
cX = int((extreme_left[0] + extreme_right[0]) / 2)
cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

# find the maximum euclidean distance between the center of the palm
# and the most extreme points of the convex hull
distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
maximum_distance = distance[distance.argmax()]

# calculate the radius of the circle with 80% of the max euclidean distance obtained
radius = int(0.5 * maximum_distance)

# find the circumference of the circle
circumference = (2 * np.pi * radius)

# take out the circular region of interest which has 
# the palm and the fingers
circular_roi = np.zeros(binary.shape[:2], dtype="uint8")

# draw the circular ROI
cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

# take bit-wise AND between thresholded hand using the circular ROI as the mask
# which gives the cuts obtained using mask on the thresholded hand image
circular_roi = cv2.bitwise_and(binary, binary, mask=circular_roi)
cv2.imshow("circular_roi",circular_roi)
#contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#画包围线，参数2是包围线列表，参数3是指包围线下标，如果为-1则全画
cv2.drawContours(img,[segmented],-1,(0,0,255),3)
#画凸围线
cv2.drawContours(img,hull,-1,(0,255,0),2)  


cv2.imshow("binary",binary)
cv2.imshow("img", img)  
#cv2.waitKey(1)  
#cv2.destroyAllWindows()

while(True):
    keypress = cv2.waitKey(1) & 0xFF
    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        cv2.destroyAllWindows()
        break


