import cv2
import math
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent

binary = cv2.imread(
    str(BASE / "steps" / "clock8-10-10" / "binary_image.png"),
    cv2.IMREAD_GRAYSCALE
)

# Crop the detected clock face
cx, cy = 113, 68
radius = 26

crop = binary[
    cy-radius:cy+radius,
    cx-radius:cx+radius
]

# Upscale the binary image
scale = 8

enlarged = cv2.resize(
    crop,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_NEAREST
)

cv2.imwrite(
    str(BASE / "test_small_binary.png"),
    enlarged
)

# Detect short hand segments
lines = cv2.HoughLinesP(
    enlarged,
    1,
    np.pi / 180,
    threshold=20,
    minLineLength=40,
    maxLineGap=16
)

display = cv2.cvtColor(
    enlarged,
    cv2.COLOR_GRAY2BGR
)

if lines is not None:

    print(f"Detected lines: {len(lines)}")

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        cv2.line(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

else:
    print("No lines detected.")

cv2.circle(
    display,
    (radius * scale, radius * scale),
    4,
    (0, 0, 255),
    -1
)

cv2.imwrite(
    str(BASE / "test_small_lines.png"),
    display
)

# Group segments by their angular direction
center = np.array(
    [radius * scale, radius * scale],
    dtype=float
)

groups = []

if lines is not None:

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        p = np.array([x1, y1], dtype=float)
        q = np.array([x2, y2], dtype=float)

        # Select the endpoint farthest from the center
        distance_p = np.linalg.norm(p - center)
        distance_q = np.linalg.norm(q - center)

        tip = p if distance_p > distance_q else q

        dx, dy = tip - center

        angle = math.degrees(
            math.atan2(dx, -dy)
        ) % 360

        radial_length = np.linalg.norm(tip - center)

        assigned = False

        for group in groups:

            difference = abs(
                (angle - group[0][0] + 180) % 360 - 180
            )

            if difference <= 15:

                group.append((angle, radial_length))
                assigned = True
                break

        if not assigned:
            groups.append([(angle, radial_length)])

# Analyze the detected groups
for i, group in enumerate(groups):

    angles = np.radians([v[0] for v in group])

    mean_angle = math.degrees(
        math.atan2(
            np.mean(np.sin(angles)),
            np.mean(np.cos(angles))
        )
    ) % 360

    length = max(v[1] for v in group)

    print(
        f"Group {i+1}: "
        f"Angle={mean_angle:.2f}, "
        f"Length={length / scale:.2f}px, "
        f"Segments={len(group)}"
    )

# Calculate hand thickness
thickness_map = cv2.distanceTransform(
    enlarged,
    cv2.DIST_L2,
    5
)

hand_candidates = []

for i, group in enumerate(groups):

    angles = np.radians([v[0] for v in group])

    angle = math.atan2(
        np.mean(np.sin(angles)),
        np.mean(np.cos(angles))
    )

    direction = np.array([
        math.sin(angle),
        -math.cos(angle)
    ])

    perpendicular = np.array([
        -direction[1],
        direction[0]
    ])

    widths = []

    for distance in np.linspace(4 * scale, 7 * scale, 20):

        point = center + distance * direction

        values = []

        for offset in range(-2 * scale, 2 * scale + 1):

            x, y = np.round(
                point + offset * perpendicular
            ).astype(int)

            if 0 <= x < enlarged.shape[1] and 0 <= y < enlarged.shape[0]:
                values.append(thickness_map[y, x])

        if values:
            widths.append(2 * max(values) / scale)

    width = np.median(widths) if widths else 0

    print(f"Group {i+1}: Width={width:.2f}px")


    


    hand_candidates.append({
        "angle": math.degrees(angle) % 360,
        "length": max(v[1] for v in group) / scale,
        "width": width
    })

print("Hand candidates:", hand_candidates)

# Reject short candidates
max_length = max(
    hand["length"] for hand in hand_candidates
)

valid_hands = [
    hand for hand in hand_candidates
    if hand["length"] >= max_length * 0.60
]

print(f"Valid hands: {len(valid_hands)}")

if len(valid_hands) == 2:

    # Identify the hour hand by its thickness
    valid_hands.sort(
        key=lambda hand: hand["width"],
        reverse=True
    )

    hour_angle = valid_hands[0]["angle"]
    minute_angle = valid_hands[1]["angle"]

    minute = round(minute_angle / 6) % 60

    hour = round(
        (hour_angle - minute_angle / 12) / 30
    ) % 12 or 12

    print(f"Hour angle: {hour_angle:.2f}")
    print(f"Minute angle: {minute_angle:.2f}")
    print(f"Detected time: {hour:02d}:{minute:02d}")

else:
    print("Could not identify two valid hands.")