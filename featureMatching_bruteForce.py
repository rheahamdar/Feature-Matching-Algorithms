import cv2
import numpy as np

img1 = cv2.imread("C:\\Users\\Rhea\\OneDrive\\Desktop\\research\\my_photos.jpg",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("C:\\Users\\Rhea\\OneDrive\\Desktop\\research\\my_photos1.jpg", cv2.IMREAD_GRAYSCALE)

#ORB DETECTOR
orb = cv2.ORB_create()
kp1, desc1 = orb.detectAndCompute(img1,None)
kp2, desc2 = orb.detectAndCompute(img2,None)

#brute force matching: compare each desciptor of first image with the descriptors in the second image
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)#crossCheck= true to take the best match not all the features
matches = bf.match(desc1,desc2)
print(len(matches))
matches = sorted(matches, key= lambda x:x.distance)

#the smaller the distance, the better is the match
for m in matches:
    print(m.distance)#distance between the matched descriptors(features)
"""
for d in desc1: #array of numbers that describes the features independent of lightning, rotation
    print(d)
"""
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)#only the first 50 matches to get the best ones

cv2.imshow("Img 1",img1)
cv2.imshow("Img 2",img2)
cv2.imshow("Matching result",matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()