import cv2
import numpy as np
from BoardDetection import detect_board

# Run board detection and collect its results.
#before_board = detect_board("Images/EmptyBoardTest.jpg")

# Detect the board in the photo containing the dart.
#after_board = detect_board("Images/BoardWithDarts.jpg")

#print("After bull centre:", after_board["bull_centre"])

#print("Bull centre:", before_board["bull_centre"])

#Add the picture of the empty dartboard 
before_image = cv2.imread("Images/Before.jpg", cv2.IMREAD_COLOR)

after_image = cv2.imread("Images/After.jpg", cv2.IMREAD_COLOR)


#Error handling for image loading

if before_image is None or after_image is None:
    raise ValueError("Could not load one or both images. Check the file paths.")


# Resize images to the same dimensions if they are not already the same size.
height, width = before_image.shape[:2]
scale = min(800 / width, 700 / height, 1)

# Use the same dimensions for both images.
new_size = (int(width * scale), int(height * scale))

before_image = cv2.resize(before_image, new_size)
after_image = cv2.resize(after_image, new_size)

#Show both images side by side for comparison.
Images =np.hstack((before_image, after_image))
cv2.imshow('Before and After Images', Images)






#Find the difference between the two images to find the darts.
difference = cv2.absdiff(before_image, after_image)

# Convert the difference to grayscale.
difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Mark changes above 25 as white; everything else becomes black.
_, dart_mask = cv2.threshold(
    difference_gray, 25, 255, cv2.THRESH_BINARY
)

# Find the outlines of the white shapes.
contours, _ = cv2.findContours(
    dart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Draw the largest shape on a copy of the after image.
dart_preview = after_image.copy()

if contours:
    dart_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(
        dart_preview, [dart_contour], -1, (0, 255, 0), 1
    )
        
        
    # For this photo, the tip is the lowest point on the contour.
    points = dart_contour.reshape(-1, 2)
    tip_x, tip_y = points[np.argmax(points[:, 1])]

    # Mark the candidate tip with a small red circle.
    cv2.circle(
        dart_preview, (int(tip_x), int(tip_y)),
        2, (0, 0, 255), 1
    )

    print("Candidate tip:", int(tip_x), int(tip_y))

cv2.imshow("Dart Outline", dart_preview)

cv2.imshow("Dart Mask", dart_mask)

cv2.imshow('Difference Image', difference)


#Find the darts in the image and their locations.

#find the score of each dart based on its location and the dartboard layout.

#display the score of each dart and the total score.

cv2.waitKey(0)
cv2.destroyAllWindows()