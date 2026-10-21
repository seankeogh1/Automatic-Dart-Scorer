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

#Doubles Numbers in order clockwise starting from 20
outer_numbers = [number * 2 for number in board_numbers]

#Triples Numbers in order clockwise starting from 20 
inner_numbers =[number * 3 for number in board_numbers]



selection_background = img.copy()

# Collect the outer single sections, one reference for each wedge.
outer_sections = []

for section in white_sections + black_sections:

    # Calculate the centre of this section.
    moments = cv2.moments(section)
    if moments["m00"] == 0:
        continue

    x = moments["m10"] / moments["m00"]
    y = moments["m01"] / moments["m00"]

    # Keep sections whose centre is outside the outer treble boundary.
    if cv2.pointPolygonTest(sorted_contours[2], (x, y), False) < 0:
        outer_sections.append(section)

# Find the direction from the bull centre to a sections centre.
def section_angle(section):

    # Calculate the section's centre.
    moments = cv2.moments(section)
    x = moments["m10"] / moments["m00"]
    y = moments["m01"] / moments["m00"]

    # Return its direction in radians.
    return np.arctan2(y - centre_y, x - centre_x)

# Sort the outer sections clockwise by their direction around the bull.
outer_sections = sorted(outer_sections, key=section_angle)


# Build a whole wedge mask from a detected single section.
def build_wedge(section):
            # Arrange the selected contour as a list of (x, y) points.
            points = section.reshape(-1, 2)
        
            # Calculate the angle from the bull centre to each outline point.
            angles = np.arctan2(
                points[:, 1] - centre_y,
                points[:, 0] - centre_x
            )

            # Measure boundary angles relative to this section's direction.
            reference_angle = section_angle(section)
            angles = (angles - reference_angle + np.pi) % (2 * np.pi) - np.pi
    
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
            # Measure pixel angles relative to the same direction.
            pixel_angles = (
                pixel_angles - reference_angle + np.pi
            ) % (2 * np.pi) - np.pi
    
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

# Store each scoring area's mask, score and label after calibration.
scoring_areas = []


# Create a filled mask inside a contour.
def filled_mask(contour):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask


# Create a ring between an outside and inside boundary.
def ring_mask(outside_contour, inside_contour):
    mask = filled_mask(outside_contour)
    cv2.drawContours(mask, [inside_contour], -1, 0, cv2.FILLED)
    return mask


# The double ring lies between the first two detected boundaries.
double_mask = ring_mask(sorted_contours[0], sorted_contours[1])

# The treble ring lies between the next two boundaries.
treble_mask = ring_mask(sorted_contours[2], sorted_contours[3])

# Outer singles lie between the double and treble rings.
outer_single_mask = ring_mask(sorted_contours[1], sorted_contours[2])

# Inner singles lie between the treble ring and outer bull.
inner_single_mask = ring_mask(sorted_contours[3], sorted_contours[4])

# Both single areas have the same scoring rule, so combine them.
single_mask = cv2.bitwise_or(outer_single_mask, inner_single_mask)

# Track whether a section has been selected and testing has started.
twenty_selected = False
testing = False

#Start by Adding a click to recognise 20 and then add the rest of the numbers clockwise.
def twenty_click(event, x, y, flags, param):
    global twenty_selected, testing

    # Ignore events other than a left click.
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    
     # Once testing starts, check which scoring area contains the click.
    if testing:

        # Ignore clicks on the Start Test button.
        if 10 <= x <= 175 and 10 <= y <= 50:
            return

        for area in scoring_areas:

            # A white pixel means this position is inside the area's mask.
            if area["mask"][y, x] != 0:
                print(f'{area["label"]}: {area["score"]}')
                return

        # The click was not inside any stored scoring area.
        print("Miss or unassigned area: 0")
        return
        

    # Start testing only after a section has been selected.
    if 10 <= x <= 175 and 10 <= y <= 50:
        if twenty_selected:
            testing = True
            print("20 confirmed. Testing started.")
        return

    # Find which outer single section was clicked.
    for section_index, section in enumerate(outer_sections):
        result = cv2.pointPolygonTest(section, (x, y), False)

        if result >= 0:
            twenty_selected = True

            # Start the clockwise section list at the selected 20.
            ordered_sections = (
                outer_sections[section_index:] +
                outer_sections[:section_index]
            )

            display = selection_background.copy()

            # Remove scoring areas from any previous selection.
            scoring_areas.clear()

            # Build and label every numbered wedge.
            for number, double_score, treble_score, scoring_section in zip(
                board_numbers, outer_numbers, inner_numbers, ordered_sections
            ):
                numbered_wedge = build_wedge(scoring_section)

                # Save the three scoring regions within this wedge.
                for region_mask, score, label in [
                    (single_mask, number, f"Single {number}"),
                    (double_mask, double_score, f"Double {number}"),
                    (treble_mask, treble_score, f"Treble {number}")
                ]:
                    area_mask = cv2.bitwise_and(
                        numbered_wedge, region_mask
                    )

                    scoring_areas.append({
                        "mask": area_mask,
                        "score": score,
                        "label": label
                    })

                # Draw one number label for this wedge.
                label_angle = section_angle(scoring_section)
                boundary_points = sorted_contours[0].reshape(-1, 2)

                boundary_angles = np.arctan2(
                    boundary_points[:, 1] - centre_y,
                    boundary_points[:, 0] - centre_x
                )

                angle_difference = np.abs(
                    (boundary_angles - label_angle + np.pi)
                    % (2 * np.pi) - np.pi
                )

                boundary_x, boundary_y = boundary_points[
                    np.argmin(angle_difference)
                ]

                label_x = int(boundary_x + 18 * np.cos(label_angle))
                label_y = int(boundary_y + 18 * np.sin(label_angle))

                cv2.putText(
                    display, str(number), (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA
                )

            # Outer bull: inside the outer bull boundary, excluding bullseye.
            scoring_areas.append({
                "mask": ring_mask(outer_bull, inner_bull),
                "score": 25,
                "label": "Outer bull"
            })

            # Bullseye: inside the inner bull boundary.
            scoring_areas.append({
                "mask": filled_mask(inner_bull),
                "score": 50,
                "label": "Bullseye"
            })

            # Highlight the selected 20 wedge after all areas are saved.
            wedge_mask = build_wedge(section)

            wedge_contours, _ = cv2.findContours(
                wedge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                display, wedge_contours, -1, (0, 255, 255), 2
            )

            # Draw the Start Test button.
            cv2.rectangle(
                display, (10, 10), (175, 50),
                (50, 120, 50), cv2.FILLED
            )

            cv2.putText(
                display, "Start Test", (20, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA
            )

            print(f"Saved {len(scoring_areas)} scoring areas.")
            cv2.imshow('Board Image', display)
            break


cv2.imshow('Board Image', img) #Displays the coloured image after elipses have been drawn.
cv2.setMouseCallback('Board Image', twenty_click) # Connect mouse events in this window to our function.
cv2.moveWindow('Board Image', 100, 100) #posistion the window on the screen at coordinates (100, 100)

cv2.waitKey(0)
cv2.destroyAllWindows()