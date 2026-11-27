import cv2
import numpy as np
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent

image = cv2.imread(
    str(BASE / "steps" / "clock15-12-00" / "hands_image.png"),
    cv2.IMREAD_GRAYSCALE
)

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Remove numbers and peripheral elements
inner_mask = np.zeros_like(binary)

cv2.circle(
    inner_mask,
    (146, 147),
    int(145 * 0.65),
    255,
    -1
)

binary = cv2.bitwise_and(binary, inner_mask)

# Detect connected components
num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(binary)
)

# Select the largest component, excluding background
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

# Extract component pixels
ys, xs = np.where(labels == largest)

points = np.column_stack((xs, ys)).astype(np.float32)

# Fit the main axis
vx, vy, x, y = cv2.fitLine(
    points, cv2.DIST_L2, 0, 0.01, 0.01
).flatten()

# Clock center from geometric detection
center = np.array([146, 147])
radius = 145

direction = np.array([vx, vy])
origin = np.array([x, y])

# Project the center onto the main axis
pivot = origin + np.dot(center - origin, direction) * direction

# Isolate the main connected component
component = np.zeros_like(binary)
component[labels == largest] = 255

# Calculate local thickness
distance_map = cv2.distanceTransform(
    component,
    cv2.DIST_L2,
    5
)

# Locate the thickest part of the component
_, max_thickness, _, thickest_point = cv2.minMaxLoc(
    distance_map
)

# Estimate the mechanical pivot
pivot = np.array(thickest_point, dtype=float)

print(f"Maximum thickness: {max_thickness:.2f}")
print(f"Thickness-based pivot: {pivot}")

# Find the longest extension from the pivot
projections = (points - pivot) @ direction

if abs(projections.min()) > abs(projections.max()):
    direction = -direction

# Calculate clockwise angle
angle = math.degrees(
    math.atan2(direction[0], -direction[1])
) % 360

print(f"Estimated pivot: {pivot}")
print(f"Main angle: {angle:.2f}")

# Find the nearest theoretical overlapping position
overlap_step = 360 / 11

overlap_angle = (
    round(angle / overlap_step) * overlap_step
) % 360

overlap_error = abs(
    (angle - overlap_angle + 180) % 360 - 180
)

print(f"Overlap error: {overlap_error:.2f}")

# Evaluate component geometry
areas = sorted(
    stats[1:, cv2.CC_STAT_AREA],
    reverse=True
)

dominance = areas[0] / max(areas[1], 1) if len(areas) > 1 else float("inf")

perpendicular = np.array([-direction[1], direction[0]])

length = np.ptp((points - pivot) @ direction)
width = np.ptp((points - pivot) @ perpendicular)

elongation = length / max(width, 1)

pivot_distance = np.linalg.norm(pivot - center)

print(f"Dominance: {dominance:.2f}")
print(f"Elongation: {elongation:.2f}")
print(f"Pivot distance: {pivot_distance:.2f}")

# Normalize the pivot distance
relative_pivot_distance = pivot_distance / radius

# Evaluate overlapping hand confidence
overlap_detected = (
    dominance >= 5
    and elongation >= 4
    and relative_pivot_distance <= 0.30
    and overlap_error <= 2
)

if overlap_detected:

    minute = round(overlap_angle / 6) % 60

    hour = round(
        (overlap_angle - minute * 0.5) / 30
    ) % 12 or 12

    print(f"Corrected angle: {overlap_angle:.2f}")
    print(f"Possible overlapping hands: {hour:02d}:{minute:02d}")

else:
    print("Overlapping hands not confirmed.")

# Visualisation
display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

display[labels == largest] = (0, 255, 0)

tip = pivot + direction * 70

cv2.arrowedLine(
    display,
    tuple(pivot.astype(int)),
    tuple(tip.astype(int)),
    (255, 0, 0),
    2
)

cv2.circle(
    display, tuple(pivot.astype(int)),
    4, (0, 0, 255), -1
)

cv2.imwrite(str(BASE / "test_overlapping.png"), display)

# Detect the inner decorative circle
original_binary = cv2.imread(
    str(BASE / "steps" / "clock15-12-00" / "binary_image.png"),
    cv2.IMREAD_GRAYSCALE
)

circles = cv2.HoughCircles(
    original_binary,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=100,
    param2=20,
    minRadius=int(radius * 0.30),
    maxRadius=int(radius * 0.65)
)

circle_display = cv2.cvtColor(
    original_binary,
    cv2.COLOR_GRAY2BGR
)

if circles is not None:
    for x, y, r in np.round(circles[0]).astype(int):

        print(f"Inner circle: ({x}, {y}), radius={r}")

        cv2.circle(
            circle_display,
            (x, y), r, (0, 255, 0), 2
        )

        cv2.circle(
            circle_display,
            (x, y), 4, (0, 0, 255), -1
        )

else:
    print("No inner circle detected.")

cv2.imwrite(
    str(BASE / "test_inner_circle.png"),
    circle_display
)

# Analyze the hand direction at different distances
ys, xs = np.where(labels == largest)

dx = xs - pivot[0]
dy = ys - pivot[1]

distances = np.hypot(dx, dy)

for low, high in [(0.20, 0.35),
                  (0.35, 0.50),
                  (0.50, 0.65)]:

    valid = (
        (distances >= radius * low)
        & (distances < radius * high)
    )

    if np.any(valid):

        mean_x = np.mean(dx[valid])
        mean_y = np.mean(dy[valid])

        local_angle = math.degrees(
            math.atan2(mean_x, -mean_y)
        ) % 360

        print(
            f"Zone {low:.2f}-{high:.2f}: "
            f"angle={local_angle:.2f}"
        )