# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:13:27 2021

@author: Sonu
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('scene2.png',0)          #QueryImage
img2 = cv2.imread('scene2_obj6.png',0) #TrainImage

#Create a Descriptor object
sift = cv2.xfeatures2d.SIFT_create()

#Extract the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

#Match object using FLANN based matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)
#Finding the top two matches for each descriptor using KNN match
matches = flann.knnMatch(des1,des2,k=2)

#Storing all the good matches as per Lowe's ratio test
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
              
#Homography is possible if atleast 4 sets of match are found        
if len(good)>MIN_MATCH_COUNT:
    #Finding the keypoints for good matches only
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    #Finding the homography between keypoints of matching pair sets using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    #Perforimg a perspective transform on the query image
    dst = cv2.perspectiveTransform(pts,M)
    
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None        

draw_params = dict(matchColor = (0,255,0), #Drawing matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, #Drawing only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray')
plt.show()    