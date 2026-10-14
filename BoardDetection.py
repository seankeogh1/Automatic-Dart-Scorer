import cv2
import numpy as np

img = cv2.imread('Images/EmptyBoardTest.jpg', cv2.IMREAD_COLOR) #Load the Dartboard image, IMREAD_COLOR loads the image in coloR, The image data is stored in img

height, width = img.shape[:2] #img.shape returns a tuple containing the dimensions of the image. The first two elements of this tuple represent the height and width of the image, respectively. The [:2] slice is used to extract only these two values, which are then assigned to the variables height and width. 
scale = min(800 / width, 700 / height) # calculate the scaling factor to resize the image while maintaining its aspect ratio.
img = cv2.resize(img, (int(width * scale), int(height * scale))) #multiply both dimentions by the same scale to keep the proportions of the image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # making a grayscale version of the image for scoring zones as they are black and white.
edges = cv2.Canny(gray, 50, 150) # detects edges where brightness changes alot 


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert the resized image from BGR color to HSV , makes a seperate image called  hsv img stays in BGR colour.

green_mask = cv2.inRange(hsv, (35, 80, 50), (90, 255, 255)) # Create a mask selecting pixels within a green HSV range. Pixels in that range will be set to 255 white and others will be set to black 0.

red_mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)) # Some reds fall in the lower end of the hue spectrum
red_mask2 = cv2.inRange(hsv, (170, 80, 50), (179, 255, 255)) # others Fll in the upper end of the hue spectrum. So I create two masks for red and combine them.
red_mask = cv2.bitwise_or(red_mask1, red_mask2) #combining the two red masks into one.
colour_mask = cv2.bitwise_or(green_mask, red_mask) #combining the green and red masks into one.

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)) #creating a small elipsed shaped kernel 9x9 grid , used to proccess and image
board_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel) #apply morphological closing to the color mask, this gets rid of the gaps caused by the wire on board.

contours, hierarchy = cv2.findContours(
    board_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE    # this finds finds the contours in the board mask, cv2.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours, cv2.CHAIN_APPROX_NONE stores all the contour points without any compression.
)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True) #sorts the contours eclosed by each boundry by area 

# check at least onne contour was found and that it has enough points to fit an ellipse
if len(sorted_contours) >= 1:
    outer_double = sorted_contours[0]

    if len(outer_double) >= 5:
        outer_double_ellipse = cv2.fitEllipse(outer_double) #Calculate the elipse that best fits the outer contour
        cv2.ellipse(img, outer_double_ellipse, (255, 0, 0), 2) #draw the elipse on the image in blue with a thickness of 2 pixels

# check at least two contours were found and that the second one has enough points to fit an ellipse
if len(sorted_contours) >= 2:
    # 1 select the second largest contour (the inner double)
    inner_double = sorted_contours[1]

    if len(inner_double) >= 5:
        inner_double_ellipse = cv2.fitEllipse(inner_double) #Calculate the elipse that best fits the inner contour
        cv2.ellipse(img, inner_double_ellipse, (0, 255, 255), 2) #draw the elipse on the image in cyan with a thickness of 2 pixels

if len(sorted_contours) >= 3:
    # 1 select the second largest contour (the inner double)
    inner_double = sorted_contours[2]

    if len(inner_double) >= 5:
        inner_double_ellipse = cv2.fitEllipse(inner_double) #Calculate the elipse that best fits the inner contour
        cv2.ellipse(img, inner_double_ellipse, (255, 0, 0), 2) #draw the elipse on the image in blue with a thickness of 2 pixels

if len(sorted_contours) >= 4:
    # 1 select the second largest contour (the inner double)
    inner_double = sorted_contours[3]

    if len(inner_double) >= 5:
        inner_double_ellipse = cv2.fitEllipse(inner_double) #Calculate the elipse that best fits the inner contour
        cv2.ellipse(img, inner_double_ellipse, (0, 255, 255), 2) #draw the elipse on the image in cyan with a thickness of 2 pixels

if len(sorted_contours) >= 5:
    # 1 select the second largest contour (the inner double)
    inner_double = sorted_contours[4]

    if len(inner_double) >= 5:
        inner_double_ellipse = cv2.fitEllipse(inner_double) #Calculate the elipse that best fits the inner contour
        cv2.ellipse(img, inner_double_ellipse, (255, 0, 0), 2) #draw the elipse on the image in blue with a thickness of 2 pixels

# Find red shapes without closing the gaps around the inner bull
red_contours, _ = cv2.findContours(
    red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

if len(sorted_contours) >= 5:
    outer_bull = sorted_contours[4]
    bull_candidates = []

    for contour in red_contours:
        if len(contour) >= 5:
            candidate_ellipse = cv2.fitEllipse(contour)
            centre = candidate_ellipse[0]

            # Only accept red shapes whose centre is inside the outer bull
            if cv2.pointPolygonTest(outer_bull, centre, False) >= 0:
                bull_candidates.append(contour)

    if bull_candidates:
        inner_bull = max(bull_candidates, key=cv2.contourArea)
        inner_bull_ellipse = cv2.fitEllipse(inner_bull)

        # Draw the inner bull boundary in yellow
        cv2.ellipse(img, inner_bull_ellipse, (0, 255, 255), 2)

# Continue only if the five main board contours were found.
if len(sorted_contours) >= 5:

    # Create a blank mask to mark where the single-scoring sections are.
    singles_mask = np.zeros(gray.shape, dtype=np.uint8)

    # Fill inside the inner double boundary, excluding doubles and outside areas.
    cv2.drawContours(
        singles_mask, [sorted_contours[1]], -1, 255, cv2.FILLED
    )

    # Remove everything inside the outer treble boundary.
    # This leaves the outer single sections.
    cv2.drawContours(
        singles_mask, [sorted_contours[2]], -1, 0, cv2.FILLED
    )

    # Add back the area inside the inner treble boundary.
    # This adds the inner single sections without adding the treble ring.
    cv2.drawContours(
        singles_mask, [sorted_contours[3]], -1, 255, cv2.FILLED
    )

    # Remove the bull area from the mask.
    cv2.drawContours(
        singles_mask, [sorted_contours[4]], -1, 0, cv2.FILLED
    )

    # Create a small kernel for cleaning up the masks.
    section_kernel = np.ones((3, 3), dtype=np.uint8)

    # Move the mask slightly away from the ring boundaries.
    singles_mask = cv2.erode(
        singles_mask, section_kernel, iterations=1
    )

    # Blur the clean grayscale image to smooth out small surface details.
    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Select obvious white areas using their brightness.
    white_mask = cv2.inRange(blurred_gray, 155, 255)

    # Select obvious black areas using their brightness.
    # Values between 105 and 155 are ignored to avoid uncertain pixels.
    black_mask = cv2.inRange(blurred_gray, 0, 105)

    # Keep white pixels only within the single-scoring areas.
    white_mask = cv2.bitwise_and(white_mask, singles_mask)

    # Keep black pixels only within the single-scoring areas.
    black_mask = cv2.bitwise_and(black_mask, singles_mask)

    # Remove small specks and thin connections from the white mask.
    white_mask = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, section_kernel
    )

    # Remove small specks and thin connections from the black mask.
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_OPEN, section_kernel
    )

    # Find contours around the white sections.
    white_contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find contours around the black sections.
    black_contours, _ = cv2.findContours(
        black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Reject tiny contours based on the size of the detected board.
    min_section_area = cv2.contourArea(sorted_contours[0]) * 0.001

    # Save the white section contours that are large enough.
    white_sections = [
        contour for contour in white_contours
        if cv2.contourArea(contour) > min_section_area
    ]

    # Save the black section contours that are large enough.
    black_sections = [
        contour for contour in black_contours
        if cv2.contourArea(contour) > min_section_area
    ]

    # Draw cyan outlines around white sections on your existing colour image.
    cv2.drawContours(img, white_sections, -1, (255, 255, 0), 1)

    # Draw magenta outlines around black sections on the same image.
    cv2.drawContours(img, black_sections, -1, (255, 0, 255), 1)

cv2.imshow('Board Image', img) #Displays the coloured image after elipses have been drawn.
cv2.moveWindow('Board Image', 100, 100) #posistion the window on the screen at coordinates (100, 100)

cv2.waitKey(0)
cv2.destroyAllWindows()