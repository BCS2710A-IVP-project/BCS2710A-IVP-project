import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent

image = cv2.imread(
    str(BASE / "clocks" / "clock15-12-00.jpg")
)

# Approximate cardinal points on the outer dial
source_points = np.float32([
    [170, 58],    # XII
    [244, 135],   # III
    [171, 250],   # VI
    [77, 187]     # IX
])

# Corresponding points on a circular clock
target_points = np.float32([
    [150, 50],
    [250, 150],
    [150, 250],
    [50, 150]
])

# Calculate the projective transformation
matrix = cv2.getPerspectiveTransform(
    source_points,
    target_points
)

rectified = cv2.warpPerspective(
    image,
    matrix,
    (300, 300)
)

# Display the selected reference points
display = image.copy()

for point in source_points:
    cv2.circle(
        display,
        tuple(point.astype(int)),
        4,
        (0, 0, 255),
        -1
    )

cv2.imwrite(
    str(BASE / "test_perspective_points.png"),
    display
)

cv2.imwrite(
    str(BASE / "test_perspective.png"),
    rectified
)

print("Perspective correction completed.")