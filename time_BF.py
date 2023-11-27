import cv2
import time

# Load the video stream
cap = cv2.VideoCapture('C:\\Users\\Rhea\\OneDrive\\Desktop\\research\\video_research.mp4') 

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# Initialize ORB detector and brute-force matcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Read the first frame
ret, prev_frame = cap.read()

# Start the timer
start_time = time.time()

while True:
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(frame, None)

    # Match descriptors using brute force
    matches = bf.match(des1, des2)

    # Draw the matches (optional)
    img_matches = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the resulting image
    cv2.imshow('Matches', img_matches)

    # Update the previous frame
    prev_frame = frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time for the Brute Force approach: {elapsed_time} seconds")

cap.release()

# Initialize FLANN-based matcher
cap = cv2.VideoCapture('C:\\Users\\Rhea\\OneDrive\\Desktop\\research\\video_research.mp4') 
orb = cv2.ORB_create()
index_params = dict(algorithm=6,trees=5) #kd trees
search_params = dict(checks=50)  # Higher values lead to more accurate but slower matching
flann = cv2.FlannBasedMatcher(index_params, search_params)

ret, prev_frame = cap.read()

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(frame, None)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for _ in range(len(matches))]

    # Ratio test as per Lowe's paper
    for i, match in enumerate(matches):
        if len(match)>=2:
            m,n = match
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

    # Draw matches
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
    img_matches = cv2.drawMatchesKnn(prev_frame, kp1, frame, kp2, matches, None, **draw_params)

    cv2.imshow('Webcam Matches', img_matches)

    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed Time for FLANN-based matcher: {elapsed_time} seconds")

cap.release()
cv2.destroyAllWindows()