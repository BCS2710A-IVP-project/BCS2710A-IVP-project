import cv2
import numpy as np
from pathlib import Path

# Information to get the images for the presentation
BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'steps'
OUTPUT.mkdir(exist_ok=True)

# Save the images
def save(name, img):
    path = OUTPUT / f"{name}.png"
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Cannot save image: {path}")

# Load the image
image = cv2.imread(str(BASE / "clock.jpg"))
if image is None:
    raise FileNotFoundError(f"Cannot open image: {BASE / 'clock.jpg'}")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
save('grayscaled_image', gray_image)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
save('blurred_image', blurred_image)

# Find edges in the image
edges_display = cv2.Canny(blurred_image, 50, 150)
save('edges_display', edges_display)

# Now, we have the base for finding all the information we need

# Find circles 
h, w = edges_display.shape
circles = cv2.HoughCircles(edges_display, cv2.HOUGH_GRADIENT, dp=1.2, 
                           minDist=min(h, w) // 2, 
                           param1=120, param2=35, 
                           minRadius=max(10, min(h, w) // 6), maxRadius=min(h, w) // 2)
if circles is None:
    raise RuntimeError("No clock circle detected.")

circles_display = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)

for x, y, r in np.round(circles[0]).astype(int):
    cv2.circle(circles_display, (int(x), int(y)), int(r), (0, 255, 0), 2)

save('circles_display', circles_display)

# Find the center
candidates = np.round(circles[0]).astype(int)
cx, cy, radius = min(candidates, key=lambda c: abs(c[0]-w/2)+abs(c[1]-h/2)-0.25*c[2])
center = (int(cx), int(cy))

cv2.circle(circles_display, center, int(radius), (0, 0, 255), 3)
cv2.circle(circles_display, center, 5, (0, 0, 255), -1)
save('center_display', circles_display)

# Detect the potential hands of the clock
lines_display = circles_display.copy()

lines = cv2.HoughLinesP(edges_display, 1, np.pi / 180, threshold=45,
    minLineLength=max(12, min(h, w) // 9), maxLineGap=12)

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        p = np.array([x1, y1], dtype=float)
        v = np.array([x2-x1, y2-y1], dtype=float)
        t = np.clip(np.dot(np.array(center)-p, v) / max(np.dot(v, v), 1), 0, 1)

        distance = np.linalg.norm(p + t*v - np.array(center))

        if distance > max(12, radius * 0.12):
            continue

        cv2.line(lines_display, (x1, y1), (x2, y2), (255, 0, 0), 2)

else:
    print("No lines detected.")

save('lines_display', lines_display)

# Detect small and long hand 

# Calculate angles

