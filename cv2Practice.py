# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:43:12 2019

@author: Sincereedu
"""

import cv2  
import numpy as np
import random
#img=np.ones((250,250))*255
#img=img.astype('uint8')
##img[::2,:]=0
#img[:120,:]=128
#ret, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY) 
#cv2.imshow("white",binary)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img=[random.randrange(0,256) for _ in range(250*250)]
img=np.array(img)
img=img.reshape((250,250))
img=img.astype('uint8')
ret, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY) 
cv2.imshow("img",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()