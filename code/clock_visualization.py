import cv2
import numpy as np
from pathlib import Path
import math 

# Information to get the images for the presentation
BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'steps'
OUTPUT.mkdir(exist_ok=True)

# Save the images
def save(name, img):
    path = OUTPUT / f"{name}.png"
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Cannot save image: {path}")
    else:
        print(f"{path} saved successfully")
    return path

# Convert points from which lines are drawn to vectors
def vector(x1,y1,x2,y2):
    vector_x = (x2 - x1)
    vector_y = (y2 - y1)
    return (vector_x, vector_y)

# Calculate length of vectors to find longer and shorter hand of the clock
def vector_length(x,y):
    x = abs(x)
    y = abs(y)
    return math.sqrt((x ** 2) + (y ** 2))

# Find only two vectors: one for longer hand, one for shorter hand
def vector_simplification(vectors):
    simplified = []
    simplified_angles = []
    for vector in sorted(vectors, key=lambda item: item[2], reverse=True):
        angle = math.degrees(math.atan2(vector[1], vector[0])) % 180
        has_similar_vector = any(
            abs((angle - kept_angle + 90) % 180 - 90) <= 10
            for kept_angle in simplified_angles
        )
        if not has_similar_vector:
            simplified.append(vector)
            simplified_angles.append(angle)
            if len(simplified) == 2:
                break
    if simplified[0][2] >= simplified[1][2]:
        return simplified[0], simplified[1]
    else:
        return simplified[1], simplified[0]

# Calculate the angle against the reference line (ie. line from number twelve to the clock center)
def angle(vector, center):
    if len(vector) >= 7:
        endpoint_one = (vector[3], vector[4])
        endpoint_two = (vector[5], vector[6])
        distance_one = vector_length(endpoint_one[0] - center[0], endpoint_one[1] - center[1])
        distance_two = vector_length(endpoint_two[0] - center[0], endpoint_two[1] - center[1])
        if distance_one >= distance_two:
            tip = endpoint_one
        else:
            tip = endpoint_two
        vector_x = tip[0] - center[0]
        vector_y = tip[1] - center[1]
    else:
        vector_x = vector[0]
        vector_y = vector[1]
    reference_x = 0
    reference_y = -1
    dot = reference_x * vector_x + reference_y * vector_y
    determinant = reference_x * vector_y - reference_y * vector_x

    clockwise_angle = math.degrees(math.atan2(determinant, dot)) % 360
    return clockwise_angle

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
simplified_lines_display = circles_display.copy()

lines = cv2.HoughLinesP(edges_display, 1, np.pi / 180, threshold=45,
    minLineLength=max(12, min(h, w) // 9), maxLineGap=12)

vectors = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        p = np.array([x1, y1], dtype=float)
        v = np.array([x2-x1, y2-y1], dtype=float)
        t = np.clip(np.dot(np.array(center)-p, v) / max(np.dot(v, v), 1), 0, 1)

        distance = np.linalg.norm(p + t*v - np.array(center))

        if distance > max(12, radius * 0.12):
            continue

        cv2.line(lines_display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        dx, dy = vector(x1, y1, x2, y2)
        length = vector_length(dx, dy)
        vectors.append([dx, dy, length, x1, y1, x2, y2])

    save('lines_display', lines_display)
    # Detect small and long hand 
    long_hand, short_hand = vector_simplification(vectors)
    cv2.line(simplified_lines_display, (long_hand[3],long_hand[4]), (long_hand[5], long_hand[6]), (255, 0, 0), 2)
    cv2.line(simplified_lines_display, (short_hand[3],short_hand[4]), (short_hand[5], short_hand[6]), (255, 0, 0), 2)
    save('simplified_lines_display', simplified_lines_display)
    # Calculate angles
    minute_angle = angle(long_hand, center)
    hour_angle = angle(short_hand, center)
    hour = str(int(hour_angle/30))
    if len(hour) == 1:
        hour = "0" + hour
    minute = str(round(minute_angle/6))
    if len(minute) == 1:
        minute = "0" + minute
    print(f"Time on the clock: {hour}:{minute}")
else:
    print("No lines detected.")


