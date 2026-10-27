import cv2
import numpy as np
from BoardDetection import detect_board

# Load the board images before and after adding a dart.
before_image = cv2.imread("Images/Before.jpg", cv2.IMREAD_COLOR)
after_image = cv2.imread("Images/After.jpg", cv2.IMREAD_COLOR)
# Check that both images loaded.
if before_image is None or after_image is None:
    raise ValueError("Could not load one or both images. Check the file paths.")


# Resize both images using the same dimensions.
height, width = before_image.shape[:2]
scale = min(1600 / width, 1000 / height, 1)
new_size = (int(width * scale), int(height * scale))

before_image = cv2.resize(before_image, new_size)
after_image = cv2.resize(after_image, new_size)


# Find changes between corresponding pixels.
difference = cv2.absdiff(before_image, after_image)
difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Keep changes above the threshold as white pixels.
_, dart_mask = cv2.threshold(
    difference_gray, 20, 255, cv2.THRESH_BINARY
)

# Preserve the mask before closing for comparison.
raw_dart_mask = dart_mask.copy()
# Join small gaps between the barrel and point.
kernel = np.ones((3, 3), dtype=np.uint8)
dart_mask = cv2.morphologyEx(
    dart_mask, cv2.MORPH_CLOSE, kernel
)


# Find outlines of the white shapes in the mask.
contours, _ = cv2.findContours(
    dart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Draw results on a copy, preserving the after image.
dart_preview = after_image.copy()

if contours:
    # For this test, treat the largest changed shape as the dart.
    dart_contour = max(contours, key=cv2.contourArea)

    #cv2.drawContours(dart_preview, [dart_contour], -1, (0, 255, 0), 1)

    # For this dart orientation, use the lowest contour point as the tip.
    points = dart_contour.reshape(-1, 2)

    # Estimate a straight line through the dart contour.
    vx, vy, x0, y0 = cv2.fitLine(
        points, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()

    # Find how far the contour extends along that line.
    distances = (
        (points[:, 0] - x0) * vx +
        (points[:, 1] - y0) * vy
    )

    start = (
        int(x0 + distances.min() * vx),
        int(y0 + distances.min() * vy)
    )

    end = (
        int(x0 + distances.max() * vx),
        int(y0 + distances.max() * vy)
    )

    # Draw the estimated direction in blue.
    #cv2.line(dart_preview, start, end, (255, 0, 0), 1)

    # Measure each contour point's perpendicular distance from the line.
    distance_from_line = np.abs(
        (points[:, 0] - x0) * vy -
        (points[:, 1] - y0) * vx
    )

    # Keep only points close to the dart's centre line.
    tip_candidates = points[distance_from_line <= 3]

    if len(tip_candidates) == 0:
        raise ValueError("No contour points found close to the dart line.")

    # For this downward-pointing dart, choose the lowest remaining point.
    tip_x, tip_y = tip_candidates[np.argmax(tip_candidates[:, 1])]

    # Draw a small red circle around the candidate tip.
    cv2.circle(
        dart_preview, (int(tip_x), int(tip_y)),
        2, (0, 0, 255), 1
    )

    print("Candidate tip:", int(tip_x), int(tip_y))



comparison = np.hstack((before_image, after_image))
cv2.imshow("Dart Mask", raw_dart_mask)
cv2.imshow("Dart Detection", dart_preview)

# Keep the windows open until a key is pressed.
cv2.waitKey(0)
cv2.destroyAllWindows()