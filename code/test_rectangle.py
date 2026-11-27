import cv2
import numpy as np
from pathlib import Path

from clock_visualization import ClockVisualization

BASE = Path(__file__).resolve().parent

def rectangle_score(rectangle, distance_map):

    x1, y1, x2, y2 = rectangle

    xs = np.linspace(x1, x2, 100).astype(int)
    ys = np.linspace(y1, y2, 100).astype(int)

    # Evaluate each rectangle side
    top = np.mean(distance_map[y1, xs] <= 2)
    bottom = np.mean(distance_map[y2, xs] <= 2)
    left = np.mean(distance_map[ys, x1] <= 2)
    right = np.mean(distance_map[ys, x2] <= 2)

    scores = [top, bottom, left, right]

    # Penalize rectangles with missing sides
    return 0.5 * np.mean(scores) + 0.5 * min(scores)

image = cv2.imread(
    str(BASE / "clocks" / "clock14-10-09.jpg")
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)

edges = cv2.Canny(blurred, 50, 100)

# Detect straight lines
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=40,
    minLineLength=65,
    maxLineGap=15
)

display = image.copy()

horizontal_lines = []
vertical_lines = []

if lines is not None:

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx)) % 180

        # Horizontal lines
        if angle < 5 or angle > 175:

            horizontal_lines.append((x1, y1, x2, y2))

            cv2.line(
                display, (x1, y1), (x2, y2),
                (0, 255, 0), 1
            )

        # Vertical lines
        elif abs(angle - 90) < 5:

            vertical_lines.append((x1, y1, x2, y2))

            cv2.line(
                display, (x1, y1), (x2, y2),
                (255, 0, 0), 1
            )

print("Horizontal lines:", len(horizontal_lines))
print("Vertical lines:", len(vertical_lines))

cv2.imwrite(str(BASE / "rectangle_lines.png"), display)

# Generate rectangle candidates

distance_map = cv2.distanceTransform(
    cv2.bitwise_not(edges),
    cv2.DIST_L2,
    5
)

h, w = edges.shape

x_positions = sorted(set(
    int((x1 + x2) / 2)
    for x1, y1, x2, y2 in vertical_lines
))

y_positions = sorted(set(
    int((y1 + y2) / 2)
    for x1, y1, x2, y2 in horizontal_lines
))

rectangles = []

for x1 in x_positions:
    for x2 in x_positions:

        width = x2 - x1

        if width < w * 0.20:
            continue

        for y1 in y_positions:
            for y2 in y_positions:

                height = y2 - y1

                if height < h * 0.20:
                    continue

                ratio = width / height
                area = width * height

                if not 0.4 <= ratio <= 2.5:
                    continue

                if not 0.08 <= area / (h * w) <= 0.80:
                    continue

                rectangle = (x1, y1, x2, y2)

                score = rectangle_score(
                    rectangle, distance_map
                )

                rectangles.append((score, rectangle))

# Select the best rectangle

rectangle_display = image.copy()

print("Rectangle candidates:", len(rectangles))

if rectangles:

    best_score, best_rectangle = max(
        rectangles,
        key=lambda r: r[0]
    )

    x1, y1, x2, y2 = best_rectangle

    cv2.rectangle(
        rectangle_display,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.circle(
        rectangle_display,
        (cx, cy),
        4,
        (0, 0, 255),
        -1
    )

    print("Rectangle score:", best_score)
    print("Rectangle:", best_rectangle)
    print("Center:", (cx, cy))

else:
    print("No rectangle detected.")

cv2.imwrite(
    str(BASE / "selected_rectangle.png"),
    rectangle_display
)

# Detect hands inside the selected rectangle

if rectangles:

    # Estimate center and radius
    center = (cx, cy)

    radius = min(x2 - x1, y2 - y1) // 2

    mask = np.zeros_like(edges)

    cv2.circle(
        mask,
        center,
        int(radius * 0.85),
        255,
        -1
    )

    masked_edges = cv2.bitwise_and(
        edges, edges, mask=mask
    )

    cv2.imwrite(
        str(BASE / "rectangle_mask.png"),
        mask
    )

    cv2.imwrite(
        str(BASE / "rectangle_masked_edges.png"),
        masked_edges
    )

    # Detect potential clock hands
    hand_lines = cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=max(12, int(radius * 0.15)),
        maxLineGap=8
    )

    hands_display = image.copy()

    cv2.circle(
        hands_display,
        center,
        int(radius * 0.85),
        (0, 255, 0),
        1
    )

    filtered_lines = []

    if hand_lines is not None:

        for x1, y1, x2, y2 in hand_lines.reshape(-1, 4):

            dx = x2 - x1
            dy = y2 - y1

            length = np.hypot(dx, dy)

            if length == 0:
                continue

            # Distance between center and infinite line
            distance = abs(
                dx * (center[1] - y1)
                - dy * (center[0] - x1)
            ) / length

            # Reject lines too far from the center
            if distance > max(5, radius * 0.30):
                continue

            filtered_lines.append((x1, y1, x2, y2))

            cv2.line(
                hands_display,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

        print("Detected hand lines:", len(hand_lines))
        print("Filtered hand lines:", len(filtered_lines))

    else:
        print("No hand lines detected.")

    cv2.circle(
        hands_display,
        center,
        3,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(
        str(BASE / "rectangle_filtered_hands.png"),
        hands_display
    )

# Select the two main clock hands

if rectangles and filtered_lines:

    clock = ClockVisualization(
        BASE / "logs" / "rectangle_test.txt"
    )

    # Create binary image
    _, binary_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if np.mean(binary_image[mask > 0]) > 127:
        binary_image = cv2.bitwise_not(binary_image)

    binary_image = cv2.bitwise_and(
        binary_image, binary_image, mask=mask
    )

    thickness_map = cv2.distanceTransform(
        binary_image, cv2.DIST_L2, 5
    )

    vectors = []

    for x1, y1, x2, y2 in filtered_lines:

        dx, dy = clock.vector(x1, y1, x2, y2)

        length = clock.vector_length(dx, dy)

        v = [dx, dy, length, x1, y1, x2, y2]

        width = clock.hand_width(
            v, center, thickness_map, radius
        )

        v.append(width)

        vectors.append(v)

    simplified_vectors = clock.vector_simplification(
        vectors, center
    )

    simplified_display = image.copy()

    if simplified_vectors is not None:

        long_hand, short_hand = simplified_vectors

        # Minute hand in blue
        cv2.line(
            simplified_display,
            (long_hand[3], long_hand[4]),
            (long_hand[5], long_hand[6]),
            (255, 0, 0), 2
        )

        # Hour hand in green
        cv2.line(
            simplified_display,
            (short_hand[3], short_hand[4]),
            (short_hand[5], short_hand[6]),
            (0, 255, 0), 2
        )

        cv2.circle(
            simplified_display,
            center,
            3,
            (0, 0, 255),
            -1
        )

        # Refine center using hand intersection

        p1 = np.array(long_hand[3:5], dtype=float)
        p2 = np.array(long_hand[5:7], dtype=float)

        p3 = np.array(short_hand[3:5], dtype=float)
        p4 = np.array(short_hand[5:7], dtype=float)

        d1 = p2 - p1
        d2 = p4 - p3

        matrix = np.column_stack((d1, -d2))

        if abs(np.linalg.det(matrix)) > 1e-6:

            t, s = np.linalg.solve(matrix, p3 - p1)

            intersection = p1 + t * d1

            distance = np.linalg.norm(
                intersection - np.array(center)
            )

            print("Intersection:", intersection)
            print("Distance to center:", distance)
            print("Maximum allowed distance:", radius * 0.35)

            if distance < radius * 0.35:

                print("Original center:", center)

                center = tuple(
                    np.round(intersection).astype(int)
                )

                print("Corrected center:", center)

                cv2.circle(
                    simplified_display,
                    center,
                    3,
                    (0, 255, 255),
                    -1
                )

        minute_angle = clock.angle(long_hand, center)
        hour_angle = clock.angle(short_hand, center)

        minute = round(minute_angle / 6) % 60

        hour = round(
            (hour_angle - minute_angle / 12) / 30
        ) % 12 or 12

        print("Minute angle:", minute_angle)
        print("Hour angle:", hour_angle)
        print(f"Detected time: {hour:02d}:{minute:02d}")

    else:
        print("Could not detect two distinct clock hands.")

    cv2.imwrite(
        str(BASE / "rectangle_simplified_hands.png"),
        simplified_display
    )