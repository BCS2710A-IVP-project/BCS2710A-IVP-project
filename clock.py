import cv2

# Load the image
image = cv2.imread("clock.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Find edges in the image

# Find circles 

# Find the center

# Detect the hands of the clock using Hough Line Transform

# Detect small and long hand 

# Calculate angles

