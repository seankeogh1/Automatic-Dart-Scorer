import cv2

img = cv2.imread('Images/EmptyBoardTest.jpg', cv2.IMREAD_COLOR)  #Loading the Image 
height, width = img.shape[:2]
scale = min(800 / width, 700 / height)  # Fit to scale
img = cv2.resize(img, (int(width * scale), int(height * scale))) 

cv2.imshow('Board Image', img)  # Display the image 
cv2.moveWindow('Board Image', 100, 100)  

cv2.waitKey(0)
cv2.destroyAllWindows()