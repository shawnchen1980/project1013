# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:43:12 2019

@author: Sincereedu
"""

import cv2  
import numpy as np
img=np.ones((250,250))
img[::2,:]=0

cv2.imshow("white",img)

cv2.waitKey(0)
cv2.destroyAllWindows()