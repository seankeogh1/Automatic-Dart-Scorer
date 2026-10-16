import cv2
import numpy as np

img = cv2.imread('Images/EmptyBoardTest.jpg', cv2.IMREAD_COLOR) #Load the Dartboard image, IMREAD_COLOR loads the image in coloR, The image data is stored in img

height, width = img.shape[:2] #img.shape returns a tuple containing the dimensions of the image. The first two elements of this tuple represent the height and width of the image, respectively. The [:2] slice is used to extract only these two values, which are then assigned to the variables height and width. 
scale = min(800 / width, 700 / height) # calculate the scaling factor to resize the image while maintaining its aspect ratio.
img = cv2.resize(img, (int(width * scale), int(height * scale))) #multiply both dimentions by the same scale to keep the proportions of the image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # making a grayscale version of the image for scoring zones as they are black and white.


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


for index, contour in enumerate(sorted_contours[:5]): #Loop for the 5 ring boundaries

    # Fitting an ellipse requires at least five points.
    if len(contour) >= 5:

        # Fit an ellipse around this ring boundary.
        ring_ellipse = cv2.fitEllipse(contour)

        # Alternate between blue and yellow outlines.
        colour = (255, 0, 0) if index % 2 == 0 else (0, 255, 255)

        # Draw the ellipse on the image.
        cv2.ellipse(img, ring_ellipse, colour, 2)

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
        
        centre_x, centre_y = inner_bull_ellipse[0] # Use the centre of the detected inner bull as the board centre.

    
        cv2.circle(img, (int(centre_x), int(centre_y)), 1, (255, 255, 255), -1) # Draw a small white dot so we can check its position.

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


#Dart Board Numbers in order clockwise starting from 20
board_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

selection_background = img.copy()

# Build a whole wedge mask from a detected single section.
def build_wedge(section):
            # Arrange the selected contour as a list of (x, y) points.
            points = section.reshape(-1, 2)
        
            # Calculate the angle from the bull centre to each outline point.
            angles = np.arctan2(
                points[:, 1] - centre_y,
                points[:, 0] - centre_x
            )
    
            # Estimate the two side directions of the selected 20 section.
            first_side = angles.min()
            second_side = angles.max()
    
            # Get the x and y coordinates of every pixel in the image.
            pixel_y, pixel_x = np.indices(gray.shape)
    
            # Calculate each pixel's direction from the bull centre.
            pixel_angles = np.arctan2(
                pixel_y - centre_y,
                 pixel_x - centre_x
            )
    
            # Mark pixels between the two sides of 20 as white.
            wedge_mask = (
                (pixel_angles >= first_side) &
                (pixel_angles <= second_side)
            ).astype(np.uint8) * 255
    
        
            # Create a blank mask.
            board_limit = np.zeros(gray.shape, dtype=np.uint8)
    
            # Mark everything inside the outer double boundary as white.
            cv2.drawContours(
                board_limit, [sorted_contours[0]], -1, 255, cv2.FILLED
            )
    
                # Exclude the bull from the allowed scoring area.
            cv2.drawContours(
                board_limit, [sorted_contours[4]], -1, 0, cv2.FILLED
            )
    
            # Keep only the part of our wedge inside that boundary.
            wedge_mask = cv2.bitwise_and(wedge_mask, board_limit)
            # Send the completed mask back to the code that called this function.
            return wedge_mask
    

#Start by Adding a click to recognise 20 and then add the rest of the numbers clockwise.
def twenty_click(event, x, y, flags, param):

    #Wait for a left mouse button click event
    if event == cv2.EVENT_LBUTTONDOWN:
        for section in white_sections + black_sections:
            result = cv2.pointPolygonTest(section, (x, y), False)

            if result >=0:
                wedge_mask = build_wedge(section)
                
                # Find the outline of the finished wedge mask.
                wedge_contours, _ = cv2.findContours(
                    wedge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Start from the saved image so previous wedge highlights are cleared.
                display = selection_background.copy()

                # Draw the whole wedge outline in yellow.
                cv2.drawContours(
                    display, wedge_contours, -1, (0, 255, 255), 2
                )

                cv2.imshow('Board Image', display)
                break  # Stop checking other sections once a match is found.


cv2.imshow('Board Image', img) #Displays the coloured image after elipses have been drawn.
cv2.setMouseCallback('Board Image', twenty_click) # Connect mouse events in this window to our function.
cv2.moveWindow('Board Image', 100, 100) #posistion the window on the screen at coordinates (100, 100)

cv2.waitKey(0)
cv2.destroyAllWindows()