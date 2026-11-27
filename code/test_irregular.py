import cv2
import numpy as np
from pathlib import Path
import math

from clock_visualization import ClockVisualization

BASE = Path(__file__).resolve().parent

OUTPUT = BASE / "steps" / "test_irregular"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load the irregular clock
image = cv2.imread(str(BASE / "clocks" / "clock5-01-39.jpg"))

if image is None:
    raise FileNotFoundError("Clock image not found.")

# Image preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 100)

cv2.imwrite(str(OUTPUT / "grayscale.png"), gray)
cv2.imwrite(str(OUTPUT / "blurred.png"), blurred)
cv2.imwrite(str(OUTPUT / "edges.png"), edges)

# Detect line segments without a circular mask
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=25,
    minLineLength=30,
    maxLineGap=10
)

lines_display = image.copy()
segments = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        length = math.hypot(x2 - x1, y2 - y1)

        segments.append((x1, y1, x2, y2, length))

        cv2.line(
            lines_display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

cv2.imwrite(
    str(OUTPUT / "all_lines.png"),
    lines_display
)

print(f"Detected segments: {len(segments)}")

# Find intersections between detected segments
intersections = []
h, w = image.shape[:2]

for i, first in enumerate(segments):

    p = np.array(first[:2], dtype=float)
    u = np.array(first[2:4], dtype=float) - p

    for second in segments[i + 1:]:

        q = np.array(second[:2], dtype=float)
        v = np.array(second[2:4], dtype=float) - q

        # Reject nearly parallel lines
        matrix = np.column_stack((u, -v))
        determinant = np.linalg.det(matrix)

        norm = np.linalg.norm(u) * np.linalg.norm(v)

        if norm == 0:
            continue

        if abs(determinant) / norm < 0.25:
            continue

        # Calculate intersection
        t, s = np.linalg.solve(matrix, q - p)

        # Reject intersections too far from the segments
        if not -0.15 <= t <= 1.15:
            continue

        if not -0.15 <= s <= 1.15:
            continue

        intersection = p + t * u

        x, y = intersection

        # Keep intersections within the central region
        if (
            w * 0.20 <= x <= w * 0.80
            and h * 0.20 <= y <= h * 0.80
        ):
            intersections.append(
                (round(x), round(y))
            )

# Display detected intersections
intersections_display = image.copy()

for x, y in intersections:

    cv2.circle(
        intersections_display,
        (x, y),
        3,
        (0, 0, 255),
        -1
    )

cv2.imwrite(
    str(OUTPUT / "filtered_intersections.png"),
    intersections_display
)

print(f"Detected intersections: {len(intersections)}")

# Group nearby intersections
clusters = []
cluster_distance = max(5, min(h, w) * 0.025)

for point in intersections:

    point = np.array(point, dtype=float)
    assigned = False

    for cluster in clusters:

        cluster_center = np.mean(cluster, axis=0)

        distance = np.linalg.norm(
            point - cluster_center
        )

        if distance <= cluster_distance:

            cluster.append(point)
            assigned = True
            break

    if not assigned:
        clusters.append([point])

# Keep clusters containing at least two intersections
clusters = [
    cluster for cluster in clusters
    if len(cluster) >= 2
]

print(f"Detected clusters: {len(clusters)}")

# Display intersection clusters
clusters_display = image.copy()

for i, cluster in enumerate(clusters):

    cx, cy = np.mean(cluster, axis=0)

    center = (round(cx), round(cy))

    cv2.circle(
        clusters_display,
        center,
        int(cluster_distance),
        (0, 255, 255),
        2
    )

    cv2.circle(
        clusters_display,
        center,
        3,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        clusters_display,
        f"{i + 1}: {len(cluster)}",
        (center[0] + 12, center[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )

    print(
        f"Cluster {i + 1}: "
        f"Center={center}, "
        f"Intersections={len(cluster)}"
    )

cv2.imwrite(
    str(OUTPUT / "clusters.png"),
    clusters_display
)

# Evaluate potential clock centers
pivot_candidates = []

for i, cluster in enumerate(clusters):

    center = np.mean(cluster, axis=0)

    associated_segments = []

    for segment in segments:

        p = np.array(segment[:2], dtype=float)
        q = np.array(segment[2:4], dtype=float)

        direction = q - p
        length = segment[4]

        if length == 0:
            continue

        # Distance to the actual segment
        t = np.dot(center - p, direction) / length**2
        t = np.clip(t, 0, 1)

        closest_point = p + t * direction

        distance = np.linalg.norm(
            center - closest_point
        )

        if distance <= cluster_distance:

            associated_segments.append(segment)

    # Calculate the total length of associated segments
    score = sum(
        segment[4]
        for segment in associated_segments
    )

    pivot_candidates.append(
        (score, center, associated_segments)
    )

    print(
        f"Cluster {i + 1}: "
        f"Segments={len(associated_segments)}, "
        f"Score={score:.1f}"
    )

# Select the best pivot candidate
if pivot_candidates:

    best_score, best_center, best_segments = max(
        pivot_candidates,
        key=lambda candidate: candidate[0]
    )

    center = tuple(
        np.round(best_center).astype(int)
    )

    pivot_display = image.copy()

    # Draw associated segments
    for x1, y1, x2, y2, length in best_segments:

        cv2.line(
            pivot_display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Draw selected pivot
    cv2.circle(
        pivot_display,
        center,
        12,
        (0, 255, 0),
        2
    )

    cv2.circle(
        pivot_display,
        center,
        4,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(
        str(OUTPUT / "selected_pivot.png"),
        pivot_display
    )

    print(f"Selected pivot: {center}")
    print(f"Pivot score: {best_score:.1f}")

else:
    print("No valid pivot detected.")

# Estimate a virtual clock radius
if pivot_candidates:

    radial_distances = []

    for x1, y1, x2, y2, length in best_segments:

        distance1 = math.hypot(
            x1 - center[0],
            y1 - center[1]
        )

        distance2 = math.hypot(
            x2 - center[0],
            y2 - center[1]
        )

        radial_distances.extend(
            [distance1, distance2]
        )

    max_distance = max(radial_distances)

    # Add a margin around the detected hands
    radius = max(1, round(max_distance * 1.25))

    print(f"Maximum radial distance: {max_distance:.1f}")
    print(f"Virtual radius: {radius}")

    # Display the virtual clock face
    virtual_circle_display = image.copy()

    cv2.circle(
        virtual_circle_display,
        center,
        radius,
        (0, 255, 0),
        2
    )

    cv2.circle(
        virtual_circle_display,
        center,
        4,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(
        str(OUTPUT / "virtual_circle.png"),
        virtual_circle_display
    )

    # Create a circular mask
    mask = np.zeros_like(edges)

    cv2.circle(
        mask,
        center,
        int(radius * 0.85),
        255,
        -1
    )

    masked_edges = cv2.bitwise_and(
        edges,
        edges,
        mask=mask
    )

    cv2.imwrite(
        str(OUTPUT / "virtual_mask.png"),
        mask
    )

    cv2.imwrite(
        str(OUTPUT / "virtual_masked_edges.png"),
        masked_edges
    )

    # Detect clock hands inside the virtual mask
    hand_lines = cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=15,
        minLineLength=max(12, int(radius * 0.15)),
        maxLineGap=8
    )

    hands_display = image.copy()

    if hand_lines is not None:

        for x1, y1, x2, y2 in hand_lines.reshape(-1, 4):

            cv2.line(
                hands_display,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

        print(f"Detected hand lines: {len(hand_lines)}")

    else:
        print("No hand lines detected.")

    cv2.circle(
        hands_display,
        center,
        4,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(
        str(OUTPUT / "virtual_hand_lines.png"),
        hands_display
    )

    # Split segments crossing the clock center
    split_segments = []
    split_count = 0

    center_point = np.array(center, dtype=float)

    if hand_lines is not None:

        for x1, y1, x2, y2 in hand_lines.reshape(-1, 4):

            p = np.array([x1, y1], dtype=float)
            q = np.array([x2, y2], dtype=float)

            direction = q - p
            length = np.linalg.norm(direction)

            if length == 0:
                continue

            # Project the center onto the line
            t = np.dot(
                center_point - p,
                direction
            ) / length**2

            projection = p + t * direction

            distance = np.linalg.norm(
                projection - center_point
            )

            # Check whether the segment crosses the pivot
            if (
                0.15 < t < 0.85
                and distance <= radius * 0.05
            ):

                px, py = map(int, np.round(projection))

                split_segments.append(
                    (x1, y1, px, py)
                )

                split_segments.append(
                    (px, py, x2, y2)
                )

                split_count += 1

            else:
                split_segments.append(
                    (x1, y1, x2, y2)
                )

    print(f"Segments split: {split_count}")
    print(f"Segments after splitting: {len(split_segments)}")

    # Display the split hand segments
    split_display = image.copy()

    for x1, y1, x2, y2 in split_segments:

        cv2.line(
            split_display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.circle(
            split_display,
            (x1, y1),
            3,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            split_display,
            (x2, y2),
            3,
            (0, 255, 255),
            -1
        )

    cv2.circle(
        split_display,
        center,
        4,
        (0, 0, 255),
        -1
    )

    cv2.imwrite(
        str(OUTPUT / "split_hand_lines.png"),
        split_display
    )

    # Initialize the existing clock analysis class
    clock = ClockVisualization(
        BASE / "logs" / "test_irregular.txt"
    )

    # Extract grayscale pixels inside the virtual mask
    masked_pixels = gray[mask > 0].reshape(-1, 1)

    # Calculate a local Otsu threshold
    otsu_threshold, _ = cv2.threshold(
        masked_pixels,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Apply the threshold to the grayscale image
    _, binary = cv2.threshold(
        gray,
        otsu_threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Keep only the virtual clock face
    binary = cv2.bitwise_and(
        binary,
        binary,
        mask=mask
    )

    print(f"Local Otsu threshold: {otsu_threshold:.1f}")

    # Compute hand thickness
    thickness_map = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    cv2.imwrite(
        str(OUTPUT / "virtual_binary.png"),
        binary
    )

    thickness_display = cv2.normalize(
        thickness_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(
        str(OUTPUT / "virtual_thickness.png"),
        thickness_display
    )

    # Build hand candidate vectors
    vectors = []

    for x1, y1, x2, y2 in split_segments:

        dx, dy = clock.vector(x1, y1, x2, y2)

        length = clock.vector_length(dx, dy)

        if length < radius * 0.15:
            continue

        # Distance between the pivot and the segment
        p = np.array([x1, y1], dtype=float)
        direction = np.array([dx, dy], dtype=float)

        t = np.dot(
            center_point - p,
            direction
        ) / length**2

        closest = p + np.clip(t, 0, 1) * direction

        distance = np.linalg.norm(
            closest - center_point
        )

        if distance > radius * 0.30:
            continue

        # Measure the radial extension of the candidate
        radial_length = max(
            math.hypot(
                x1 - center[0],
                y1 - center[1]
            ),
            math.hypot(
                x2 - center[0],
                y2 - center[1]
            )
        )

        # Reject short hands and counterweights
        if radial_length < radius * 0.48:
            continue

        v = [dx, dy, length, x1, y1, x2, y2]

        width = clock.hand_width(
            v,
            center,
            thickness_map,
            radius
        )

        v.append(width)

        vectors.append(v)

        print(
            f"Angle: {clock.angle(v, center):.2f}, "
            f"Length: {length:.2f}, "
            f"Radial length: {radial_length:.2f}, "
            f"Width: {width:.2f}"
        )

    print(f"Filtered vectors: {len(vectors)}")

    # Select the hour and minute hands
    simplified_vectors = clock.vector_simplification(
        vectors,
        center
    )

    if simplified_vectors is not None:

        long_hand, short_hand = simplified_vectors

        minute_angle = clock.angle(long_hand, center)
        hour_angle = clock.angle(short_hand, center)

        minute = round(minute_angle / 6) % 60

        hour = round(
            (hour_angle - minute_angle / 12) / 30
        ) % 12 or 12

        print(f"Minute angle: {minute_angle:.2f}")
        print(f"Hour angle: {hour_angle:.2f}")
        print(f"Detected time: {hour:02d}:{minute:02d}")

        simplified_display = image.copy()

        # Minute hand in blue
        cv2.line(
            simplified_display,
            tuple(long_hand[3:5]),
            tuple(long_hand[5:7]),
            (255, 0, 0),
            3
        )

        # Hour hand in green
        cv2.line(
            simplified_display,
            tuple(short_hand[3:5]),
            tuple(short_hand[5:7]),
            (0, 255, 0),
            3
        )

        cv2.circle(
            simplified_display,
            center,
            4,
            (0, 0, 255),
            -1
        )

        cv2.imwrite(
            str(OUTPUT / "virtual_simplified_hands.png"),
            simplified_display
        )

    else:
        print("Could not detect two distinct hands.")