import cv2

img = cv2.imread('Images/EmptyBoardTest.jpg', cv2.IMREAD_COLOR) #Load the Dartboard image, IMREAD_COLOR loads the image in coloR, The image data is stored in img

height, width = img.shape[:2] #img.shape returns a tuple containing the dimensions of the image. The first two elements of this tuple represent the height and width of the image, respectively. The [:2] slice is used to extract only these two values, which are then assigned to the variables height and width. 
scale = min(800 / width, 700 / height) # calculate the scaling factor to resize the image while maintaining its aspect ratio.
img = cv2.resize(img, (int(width * scale), int(height * scale))) #multiply both dimentions by the same scale to keep the proportions of the image.

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

cv2.imshow('Board Image', img) #Displays the coloured image after elipses have been drawn.
cv2.moveWindow('Board Image', 100, 100) #posistion the window on the screen at coordinates (100, 100)

cv2.waitKey(0)
cv2.destroyAllWindows()