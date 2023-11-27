import cv2
import numpy as np
import time

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def brute_force_knn(query_descriptors, data_descriptors, k=1):
    distances = [euclidean_distance(query_descriptor, data_descriptor) for query_descriptor in query_descriptors for data_descriptor in data_descriptors]
    distances = np.array(distances).reshape(len(query_descriptors), len(data_descriptors))
    indices = np.argsort(distances, axis=1)[:,:k]
    return indices

cap = cv2.VideoCapture('C:\\Users\\Rhea\\OneDrive\\Desktop\\research\\video_research.mp4')  

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

orb = cv2.ORB_create()

k_neighbors = 3

ret, prev_frame = cap.read()

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    kp_prev, des_prev = orb.detectAndCompute(prev_frame, None)
    kp_current, des_current = orb.detectAndCompute(frame, None)

    nearest_indices = brute_force_knn(des_prev, des_current, k_neighbors)

    print("Nearest Neighbors Indices:", nearest_indices)

    img_prev_with_keypoints = cv2.drawKeypoints(prev_frame, kp_prev, None, color=(0, 255, 0), flags=0)
    img_current_with_keypoints = cv2.drawKeypoints(frame, kp_current, None, color=(0, 255, 0), flags=0)

    cv2.imshow('Previous Frame with Keypoints', img_prev_with_keypoints)
    cv2.imshow('Current Frame with Keypoints', img_current_with_keypoints)

    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed Time for the FLANN Brute Force approach written by me: {elapsed_time} seconds")

cap.release()

#Brute Force from OpenCv
cap = cv2.VideoCapture('C:\\Users\\Rhea\\OneDrive\\Desktop\\research\\video_research.mp4') 

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, prev_frame = cap.read()

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(frame, None)

    matches = bf.match(des1, des2)

    img_matches = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Matches', img_matches)

    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed Time for the Brute Force approach in OpenCv: {elapsed_time} seconds")

cap.release()
cv2.destroyAllWindows()