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

# Calculate how well a circle matches the detected edges
def circle_score(x, y, r, distance_map):
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    points_x = np.round(x + r * np.cos(angles)).astype(int)
    points_y = np.round(y + r * np.sin(angles)).astype(int)

    h, w = distance_map.shape

    valid = ((points_x >= 0) & (points_x < w) & (points_y >= 0) & (points_y < h))

    distances = distance_map[points_y[valid], points_x[valid]]

    tolerance = max(3, r * 0.03)

    score = np.sum(distances <= tolerance) / 360

    return score

# Optimize the selected circle
def optimize_circle(x, y, r, distance_map):
    best = (int(x), int(y), int(r))
    best_score = circle_score(*best, distance_map)

    for step in [16, 8, 4, 2, 1]:
        improved = True

        while improved:
            improved = False
            bx, by, br = best

            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    for dr in [-step, 0, step]:
                        candidate = (bx + dx, by + dy, br + dr)

                        if candidate[2] <= 0:
                            continue

                        score = circle_score(*candidate, distance_map)

                        if score > best_score:
                            best = candidate
                            best_score = score
                            improved = True

    return best

# Measure hand width at a fixed distance from the center
def hand_width(v, center, thickness_map, radius):
    hand_angle = math.radians(angle(v, center))

    # Direction of the hand
    ux = math.sin(hand_angle)
    uy = -math.cos(hand_angle)

    # Perpendicular direction
    nx = -uy
    ny = ux

    widths = []
    h, w = thickness_map.shape

    search_range = max(3, int(radius * 0.05))

    for distance in np.linspace(radius * 0.22, radius * 0.40, 25):
        px = center[0] + distance * ux
        py = center[1] + distance * uy

        values = []

        for offset in range(-search_range, search_range + 1):
            x = int(round(px + offset * nx))
            y = int(round(py + offset * ny))

            if 0 <= x < w and 0 <= y < h:
                values.append(thickness_map[y, x])

        if values:
            widths.append(2 * max(values))

    if not widths:
        return 0

    return float(np.median(widths))

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

# Select the hour and minute hands
def vector_simplification(vectors, center):
    simplified = []
    simplified_angles = []

    # Select the two thickest hands
    for v in sorted(vectors, key=lambda item: (item[7], item[2]), reverse=True):
        hand_angle = angle(v, center)

        has_similar_vector = any(
            abs((hand_angle - kept_angle + 180) % 360 - 180) <= 10
            for kept_angle in simplified_angles
        )

        if not has_similar_vector:
            simplified.append(v)
            simplified_angles.append(hand_angle)

            if len(simplified) == 2:
                break

    if len(simplified) < 2:
        raise RuntimeError("Could not detect two distinct clock hands.")

    def hand_length(v):
        direction = angle(v, center)
        group = [u for u in vectors if abs((angle(u, center) - direction + 180) % 360 - 180) <= 10]

        return max(math.hypot(u[i]-center[0], u[i+1]-center[1]) for u in group for i in (3,5))
    
    print('Lengths',[(angle(v,center),hand_length(v),v[7]) for v in simplified])
    simplified.sort(key=hand_length, reverse=True)

    return simplified[0], simplified[1]

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
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
save('blurred_image', blurred_image)

# Find edges in the image
edges_display = cv2.Canny(blurred_image, 100, 200)
save('edges_display', edges_display)

# Find circles using adaptive detection
h, w = gray_image.shape

circles = None

for threshold in [60, 50, 40, 30, 20]:

    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(10, int(min(h, w) * 0.03)),
        param1=180,
        param2=threshold,
        minRadius=int(min(h, w) * 0.05),
        maxRadius=int(min(h, w) * 0.50)
    )

    if circles is not None:
        print(f"Circle threshold: {threshold}")
        print(f"Detected circles: {len(circles[0])}")
        break

if circles is None:
    raise RuntimeError("No clock circle detected.")

circles_display = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)

for x, y, r in np.round(circles[0]).astype(int):
    cv2.circle(circles_display, (int(x), int(y)), int(r), (0, 255, 0), 2)

save('circles_display', circles_display)

# Calculate distance to the closest edge
distance_map = cv2.distanceTransform(cv2.bitwise_not(edges_display), cv2.DIST_L2, 5)

# Find the best circle
candidates = np.round(circles[0]).astype(int)

cx, cy, radius = max(candidates, key=lambda c: circle_score(c[0], c[1], c[2], distance_map))
center = (int(cx), int(cy))

# Refine the best circle
cx, cy, radius = optimize_circle(cx, cy, radius, distance_map)

center = (int(cx), int(cy))

cv2.circle(circles_display, center, int(radius), (0, 0, 255), 3)
cv2.circle(circles_display, center, 5, (0, 0, 255), -1)
save('center_display', circles_display)

# Create a circular mask
mask = np.zeros_like(gray_image)

cv2.circle(mask, center, int(radius * 0.85), 255, -1)

save('circular_mask', mask)

# Keep only edges inside the clock
masked_edges = cv2.bitwise_and(edges_display, edges_display, mask=mask)

save('masked_edges', masked_edges)

# Create a binary image for hand width measurement
_, binary_image = cv2.threshold(
    gray_image,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

if np.mean(binary_image[mask > 0]) > 127:
    binary_image = cv2.bitwise_not(binary_image)

binary_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)

save('binary_image', binary_image)

# Detect the potential hands of the clock
lines_display = circles_display.copy()
simplified_lines_display = circles_display.copy()

# Calculate hand thickness
thickness_map = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

# Save thickness visualization
thickness_display = cv2.normalize(thickness_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

save('thickness_display', thickness_display)

# Remove thin objects
kernel_size = max(3, int(radius * 0.035))

# Ensure an odd kernel size
if kernel_size % 2 == 0:
    kernel_size += 1

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

filtered_hands = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

save('filtered_hands', filtered_hands)

# Detect edges of remaining hands
hands_edges = cv2.Canny(filtered_hands, 90, 180)
hands_edges = cv2.bitwise_and(hands_edges, hands_edges, mask=mask)

save('hands_edges', hands_edges)

# Remove thin elements, including the second hand
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

hands_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

save('hands_image', hands_image)

# Detect the remaining edges
hands_edges = cv2.Canny(hands_image, 90, 180)

save('hands_edges', hands_edges)

# Initial line detection
lines = cv2.HoughLinesP(
    masked_edges,
    1,
    np.pi / 180,
    threshold=45,
    minLineLength=max(12, int(radius * 0.25)),
    maxLineGap=12
)

# Retry line detection with less smoothing
if lines is None or len(lines) < 2:

    hand_blur = cv2.GaussianBlur(gray_image, (3, 3), 0)

    detailed_edges = cv2.Canny(hand_blur, 50, 100)

    detailed_edges = cv2.bitwise_and(detailed_edges, detailed_edges, mask=mask)

    save('detailed_edges', detailed_edges)

    lines = cv2.HoughLinesP(
        detailed_edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=max(12, int(radius * 0.15)),
        maxLineGap=8
    )

# Display all detected lines before filtering
all_lines_display = circles_display.copy()

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(all_lines_display, (x1, y1), (x2, y2), (255, 0, 0), 2)

save('all_lines_display', all_lines_display)

vectors = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        dx, dy = vector(x1, y1, x2, y2)
        length = vector_length(dx, dy)

        v = [dx, dy, length, x1, y1, x2, y2]

        width = hand_width(v, center, thickness_map, radius)

        v.append(width)

        vectors.append(v)

        print(
            f"Angle: {angle(v, center):.1f}°, "
            f"Length: {length:.1f}px, "
            f"Width: {width:.1f}px"
        )

    save('lines_display', lines_display)

    # Detect small and long hand 

    print(f"Detected lines: {len(lines)}")
    print(f"Filtered vectors: {len(vectors)}")

    long_hand, short_hand = vector_simplification(vectors, center)
    # Minute hand: blue
    cv2.line(
        simplified_lines_display,
        (long_hand[3], long_hand[4]),
        (long_hand[5], long_hand[6]),
        (255, 0, 0), 3
    )

    # Hour hand: green
    cv2.line(
        simplified_lines_display,
        (short_hand[3], short_hand[4]),
        (short_hand[5], short_hand[6]),
        (0, 255, 0), 3
    )

    save('simplified_lines_display', simplified_lines_display)

    # Calculate angles
    p = np.array(long_hand[3:5], dtype=float)
    q = np.array(short_hand[3:5], dtype=float)
    u = np.array(long_hand[5:7], dtype=float) - p
    v = np.array(short_hand[5:7], dtype=float) - q

    if abs(np.linalg.det(np.column_stack((u, -v)))) > 1e-6:
        t = np.linalg.solve(np.column_stack((u, -v)), q-p)[0]
        pivot = p+t*u

        if np.linalg.norm(pivot-np.array(center)) < radius*0.35:
            center = tuple(pivot)

    print('Pivot',center)

    minute_angle = angle(long_hand, center)
    hour_angle = angle(short_hand, center)
    hour = str(round((hour_angle - minute_angle / 12) / 30) % 12 or 12)

    if len(hour) == 1:
        hour = "0" + hour

    minute = str(round(minute_angle/6) % 60)

    if len(minute) == 1:
        minute = "0" + minute

    print(f"Time on the clock: {hour}:{minute}")
    
else:
    print("No lines detected.")
