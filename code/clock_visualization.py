import cv2
import numpy as np
from pathlib import Path
import math

class ClockVisualization:
    def __init__(self,log_file):
        # Information to get the images for the presentation
        self.base = Path(__file__).resolve().parent
        self.output = self.base / 'steps'
        self.output.mkdir(exist_ok=True)
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.photo_name = "unspecified"

    def log(self, message):
        print(f"{self.photo_name}: {message}")
        with self.log_file.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{self.photo_name}: {message}\n")

    # Save the images
    def save(self, name, img):
        path = self.output / f"{name}.png"

        if not cv2.imwrite(str(path), img):
            raise OSError(f"Cannot save image: {path}")

        self.log(f"{path} saved successfully")
        return path

    def ellipse_score(self, ellipse, distance_map):
        (cx, cy), (width, height), ellipse_angle = ellipse

        a = width / 2
        b = height / 2
        phi = np.radians(ellipse_angle)

        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)

        h, w = distance_map.shape

        tolerance = np.clip(min(a, b) * 0.015, 2, 5)

        def contour_score(scale):

            x = a * scale * np.cos(theta)
            y = b * scale * np.sin(theta)

            xs = np.round(
                cx + x * np.cos(phi) - y * np.sin(phi)
            ).astype(int)

            ys = np.round(
                cy + x * np.sin(phi) + y * np.cos(phi)
            ).astype(int)

            valid = (
                (xs >= 0) & (xs < w) &
                (ys >= 0) & (ys < h)
            )

            distances = distance_map[ys[valid], xs[valid]]

            return np.sum(distances <= tolerance) / 360

        main_score = contour_score(1.0)
        inner_score = contour_score(0.90)
        outer_score = contour_score(1.10)

        background_score = (
            inner_score + outer_score
        ) / 2

        return main_score * (1 - 0.5 * background_score)

    # Evaluate the presence of concentric ellipses
    def concentric_score(self, ellipse, ellipses):

        (cx, cy), (width, height), _ = ellipse

        radius = min(width, height) / 2
        axis_ratio = min(width, height) / max(width, height)

        count = 0

        for other in ellipses:

            if other is ellipse:
                continue

            (ox, oy), (ow, oh), _ = other

            other_radius = min(ow, oh) / 2
            other_ratio = min(ow, oh) / max(ow, oh)

            center_distance = math.hypot(
                cx - ox, cy - oy
            )

            size_ratio = other_radius / radius

            if (
                center_distance < radius * 0.10
                and 0.75 < size_ratio < 1.30
                and abs(axis_ratio - other_ratio) < 0.15
            ):
                count += 1

        return min(count / 3, 1.0)

    # Transform an elliptical clock face into a circular one
    def rectify_ellipse(self, image, ellipse):

        (ex, ey), (ew, eh), ellipse_angle = ellipse

        a = ew / 2
        b = eh / 2

        target_radius = max(a, b)

        phi = np.radians(ellipse_angle)

        rotation = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])

        scaling = np.array([
            [target_radius / a, 0],
            [0, target_radius / b]
        ])

        linear = rotation @ scaling @ rotation.T

        ellipse_center = np.array([ex, ey])

        translation = ellipse_center - linear @ ellipse_center

        matrix = np.column_stack((linear, translation))

        h, w = image.shape[:2]

        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )

    # Evaluate how well a rectangle matches the detected edges
    def rectangle_score(self, rectangle, distance_map):

        x1, y1, x2, y2 = rectangle

        xs = np.linspace(x1, x2, 100).astype(int)
        ys = np.linspace(y1, y2, 100).astype(int)

        top = np.mean(distance_map[y1, xs] <= 2)
        bottom = np.mean(distance_map[y2, xs] <= 2)
        left = np.mean(distance_map[ys, x1] <= 2)
        right = np.mean(distance_map[ys, x2] <= 2)

        scores = [top, bottom, left, right]

        return 0.5 * np.mean(scores) + 0.5 * min(scores)

    # Detect rectangular clock faces
    def detect_rectangle(self, image, edges):

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

        if lines is None:
            return None

        for x1, y1, x2, y2 in lines.reshape(-1, 4):

            dx = x2 - x1
            dy = y2 - y1

            angle = np.degrees(
                np.arctan2(dy, dx)
            ) % 180

            if angle < 5 or angle > 175:

                horizontal_lines.append(
                    (x1, y1, x2, y2)
                )

                cv2.line(
                    display,
                    (x1, y1), (x2, y2),
                    (0, 255, 0), 1
                )

            elif abs(angle - 90) < 5:

                vertical_lines.append(
                    (x1, y1, x2, y2)
                )

                cv2.line(
                    display,
                    (x1, y1), (x2, y2),
                    (255, 0, 0), 1
                )

        self.save("rectangle_lines", display)

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

        best_rectangle = None
        best_score = -1
        candidate_count = 0

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

                        score = self.rectangle_score(
                            rectangle, distance_map
                        )

                        candidate_count += 1

                        if score > best_score:
                            best_score = score
                            best_rectangle = rectangle

        self.log(
            f"Rectangle candidates: {candidate_count}"
        )

        if best_rectangle is None or best_score < 0.70:
            self.log("No valid rectangle detected.")
            return None

        x1, y1, x2, y2 = best_rectangle

        center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

        radius = min(x2 - x1, y2 - y1) // 2

        rectangle_display = image.copy()

        cv2.rectangle(
            rectangle_display,
            (x1, y1),
            (x2, y2),
            (0, 0, 255), 2
        )

        cv2.circle(
            rectangle_display,
            center,
            4,
            (0, 0, 255), -1
        )

        self.save(
            "selected_rectangle",
            rectangle_display
        )

        self.log(f"Rectangle score: {best_score:.3f}")
        self.log(f"Rectangle: {best_rectangle}")
        self.log(f"Rectangle center: {center}")

        return center, radius, rectangle_display

    # Calculate how well a circle matches the detected edges
    def circle_score(self, x, y, r, distance_map):
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
    def optimize_circle(self, x, y, r, distance_map):
        best = (int(x), int(y), int(r))
        best_score = self.circle_score(*best, distance_map)

        initial_radius = int(r)
        min_radius = max(1, int(initial_radius * 0.80))
        max_radius = int(initial_radius * 1.20)

        for step in [16, 8, 4, 2, 1]:
            improved = True

            while improved:
                improved = False
                bx, by, br = best

                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        for dr in [-step, 0, step]:
                            candidate = (bx + dx, by + dy, br + dr)

                            if not min_radius <= candidate[2] <= max_radius:
                                continue

                            score = self.circle_score(*candidate, distance_map)

                            if score > best_score:
                                best = candidate
                                best_score = score
                                improved = True

        return best

    # Measure hand width at a fixed distance from the center
    def hand_width(self, v, center, thickness_map, radius):
        hand_angle = math.radians(self.angle(v, center))

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
    def vector(self, x1, y1, x2, y2):
        vector_x = (x2 - x1)
        vector_y = (y2 - y1)

        return (vector_x, vector_y)

    # Calculate length of vectors to find longer and shorter hand of the clock
    def vector_length(self, x, y):
        x = abs(x)
        y = abs(y)

        return math.sqrt((x ** 2) + (y ** 2))

    # Select the hour and minute hands
    def vector_simplification(self, vectors, center, angle_threshold=15):
        simplified = []
        simplified_angles = []

        # Select the two thickest hands
        for v in sorted(vectors, key=lambda item: (item[7], item[2]), reverse=True):
            hand_angle = self.angle(v, center)

            has_similar_vector = any(
                self.angular_difference(hand_angle, kept_angle) <= angle_threshold
                for kept_angle in simplified_angles
            )

            if not has_similar_vector:
                simplified.append(v)
                simplified_angles.append(hand_angle)

                if len(simplified) == 2:
                    break

        if len(simplified) < 2:
            return None

        def hand_length(v):
            direction = self.angle(v, center)
            group = [
                u for u in vectors
                if abs((self.angle(u, center) - direction + 180) % 360 - 180) <= 10
            ]

            return max(
                math.hypot(u[i] - center[0], u[i + 1] - center[1])
                for u in group
                for i in (3, 5)
            )

        self.log(
            f"Lengths {[(self.angle(v, center), hand_length(v), v[7]) for v in simplified]}"
        )
        simplified.sort(key=hand_length, reverse=True)

        return simplified[0], simplified[1]

    def angular_difference(self, angle1, angle2):
        return abs((angle1 - angle2 + 180) % 360 - 180)

    # Calculate the angle against the reference line (ie. line from number twelve to the clock center)
    def angle(self, vector, center):
        if len(vector) >= 7:
            endpoint_one = (vector[3], vector[4])
            endpoint_two = (vector[5], vector[6])

            distance_one = self.vector_length(
                endpoint_one[0] - center[0],
                endpoint_one[1] - center[1]
            )
            distance_two = self.vector_length(
                endpoint_two[0] - center[0],
                endpoint_two[1] - center[1]
            )

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

    def detect_irregular_clock(self, image):
        self.log("Trying irregular clock detection.")
        # Image preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 100)

        self.save("irregular_grayscale", gray)
        self.save("irregular_blurred", blurred)
        self.save("irregular_edges", edges)

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

        self.save("irregular_all_lines", lines_display)

        self.log(f"Detected segments: {len(segments)}")

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

        self.save("irregular_filtered_intersections", intersections_display)

        self.log(f"Detected intersections: {len(intersections)}")

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

        self.log(f"Detected clusters: {len(clusters)}")

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

            self.log(
                f"Cluster {i + 1}: "
                f"Center={center}, "
                f"Intersections={len(cluster)}"
            )

        self.save("irregular_clusters", clusters_display)

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

            self.log(
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

            self.save("irregular_selected_pivot", pivot_display)

            self.log(f"Selected pivot: {center}")
            self.log(f"Pivot score: {best_score:.1f}")

        else:
            self.log("No valid pivot detected.")
            return None

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

            if not radial_distances:
                return None
            max_distance = max(radial_distances)

            # Add a margin around the detected hands
            radius = max(1, round(max_distance * 1.25))

            self.log(f"Maximum radial distance: {max_distance:.1f}")
            self.log(f"Virtual radius: {radius}")

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

            self.save("irregular_virtual_circle", virtual_circle_display)

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

            self.save("irregular_virtual_mask", mask)

            self.save("irregular_virtual_masked_edges", masked_edges)

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

                self.log(f"Detected hand lines: {len(hand_lines)}")

            else:
                self.log("No hand lines detected.")

            cv2.circle(
                hands_display,
                center,
                4,
                (0, 0, 255),
                -1
            )

            self.save("irregular_virtual_hand_lines", hands_display)

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

            self.log(f"Segments split: {split_count}")
            self.log(f"Segments after splitting: {len(split_segments)}")

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

            self.save("irregular_split_hand_lines", split_display)

            # Extract grayscale pixels inside the virtual mask
            masked_pixels = gray[mask > 0].reshape(-1, 1)

            # Calculate a local Otsu threshold
            if masked_pixels.size == 0:
                return None
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

            self.log(f"Local Otsu threshold: {otsu_threshold:.1f}")

            # Compute hand thickness
            thickness_map = cv2.distanceTransform(
                binary,
                cv2.DIST_L2,
                5
            )

            self.save("irregular_virtual_binary", binary)

            thickness_display = cv2.normalize(
                thickness_map,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            self.save("irregular_virtual_thickness", thickness_display)

            # Build hand candidate vectors
            vectors = []

            for x1, y1, x2, y2 in split_segments:

                dx, dy = self.vector(x1, y1, x2, y2)

                length = self.vector_length(dx, dy)

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

                width = self.hand_width(
                    v,
                    center,
                    thickness_map,
                    radius
                )

                v.append(width)

                vectors.append(v)

                self.log(
                    f"Angle: {self.angle(v, center):.2f}, "
                    f"Length: {length:.2f}, "
                    f"Radial length: {radial_length:.2f}, "
                    f"Width: {width:.2f}"
                )

            self.log(f"Filtered vectors: {len(vectors)}")

            # Select the hour and minute hands
            simplified_vectors = self.vector_simplification(
                vectors,
                center
            )

            if simplified_vectors is not None:

                long_hand, short_hand = simplified_vectors

                minute_angle = self.angle(long_hand, center)
                hour_angle = self.angle(short_hand, center)

                minute = round(minute_angle / 6) % 60

                hour = round(
                    (hour_angle - minute_angle / 12) / 30
                ) % 12 or 12

                self.log(f"Minute angle: {minute_angle:.2f}")
                self.log(f"Hour angle: {hour_angle:.2f}")
                self.log(f"Detected time: {hour:02d}:{minute:02d}")

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

                self.save("irregular_virtual_simplified_hands", simplified_display)
                return f"{hour:02d}:{minute:02d}"

            else:
                self.log("Could not detect two distinct hands.")
        return None

    # Detect a dominant silhouette compatible with overlapping clock hands
    def detect_overlapping_hands(self, binary, center, radius):
        if radius <= 0:
            return None

        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if count < 2:
            return None

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ys, xs = np.where(labels == largest)
        if len(xs) < 2:
            return None

        points = np.column_stack((xs, ys)).astype(np.float32)
        vx, vy, x, y = cv2.fitLine(
            points, cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        center = np.array(center, dtype=float)
        direction = np.array([vx, vy], dtype=float)
        origin = np.array([x, y], dtype=float)
        pivot = origin + np.dot(center - origin, direction) * direction
        projections = (points - pivot) @ direction
        if abs(projections.min()) > abs(projections.max()):
            direction = -direction

        angle = math.degrees(math.atan2(direction[0], -direction[1])) % 360
        overlap_step = 360 / 11
        overlap_angle = (round(angle / overlap_step) * overlap_step) % 360
        overlap_error = self.angular_difference(angle, overlap_angle)

        areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
        dominance = areas[0] / max(areas[1], 1) if len(areas) > 1 else float("inf")
        perpendicular = np.array([-direction[1], direction[0]])
        length = np.ptp((points - pivot) @ direction)
        width = np.ptp((points - pivot) @ perpendicular)
        elongation = length / max(width, 1)
        relative_pivot_distance = np.linalg.norm(pivot - center) / radius

        self.log(
            f"Overlap diagnostics: angle={angle:.2f}, error={overlap_error:.2f}, "
            f"dominance={dominance:.2f}, elongation={elongation:.2f}, "
            f"relative pivot distance={relative_pivot_distance:.3f}"
        )

        if not (
            dominance >= 5
            and elongation >= 4
            and relative_pivot_distance <= 0.30
            and overlap_error <= 2
        ):
            self.log("Overlapping hands not confirmed.")
            return None

        minute = round(overlap_angle / 6) % 60
        hour = round((overlap_angle - overlap_angle / 12) / 30) % 12 or 12
        result = f"{hour:02d}:{minute:02d}"

        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        display[labels == largest] = (0, 255, 0)
        tip = pivot + direction * max(1, np.max((points - pivot) @ direction))
        cv2.arrowedLine(display, tuple(pivot.astype(int)),
                        tuple(tip.astype(int)), (255, 0, 0), 2)
        cv2.circle(display, tuple(pivot.astype(int)), 4, (0, 0, 255), -1)
        self.save("overlapping_hands_display", display)
        self.log(f"Overlapping hands detected: {result}")
        return result

    def detect_small_clock(self, binary, center, radius):
        self.log("Trying small clock detection.")
        radius = int(radius)
        cx, cy = map(int, center)
        if radius <= 0 or radius > 40:
            return None
        h, w = binary.shape
        if cx-radius < 0 or cy-radius < 0 or cx+radius > w or cy+radius > h:
            self.log("Small clock crop exceeds image bounds.")
            return None
        crop = binary[cy-radius:cy+radius, cx-radius:cx+radius].copy()
        # Upscale the binary image
        scale = 8

        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        self.save("small_binary", enlarged)

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

            self.log(f"Detected lines: {len(lines)}")

            for x1, y1, x2, y2 in lines.reshape(-1, 4):

                cv2.line(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        else:
            self.log("No lines detected.")

        cv2.circle(
            display,
            (radius * scale, radius * scale),
            4,
            (0, 0, 255),
            -1
        )

        self.save("small_lines", display)

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

            self.log(
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

            self.log(f"Group {i+1}: Width={width:.2f}px")

            hand_candidates.append({
                "angle": math.degrees(angle) % 360,
                "length": max(v[1] for v in group) / scale,
                "width": width
            })

        self.log(f"Hand candidates: {hand_candidates}")

        # Reject short candidates
        if not hand_candidates:
            return None
        max_length = max(
            hand["length"] for hand in hand_candidates
        )

        valid_hands = [
            hand for hand in hand_candidates
            if hand["length"] >= max_length * 0.60
        ]

        self.log(f"Valid hands: {len(valid_hands)}")

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

            self.log(f"Hour angle: {hour_angle:.2f}")
            self.log(f"Minute angle: {minute_angle:.2f}")
            self.log(f"Detected time: {hour:02d}:{minute:02d}")
            selected = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
            for hand, color in zip(valid_hands, [(0, 255, 0), (255, 0, 0)]):
                a = math.radians(hand["angle"])
                tip = center + hand["length"] * scale * np.array([math.sin(a), -math.cos(a)])
                cv2.arrowedLine(selected, tuple(center.astype(int)), tuple(np.round(tip).astype(int)), color, 3)
            cv2.circle(selected, tuple(center.astype(int)), 4, (0, 0, 255), -1)
            self.save("small_simplified_hands", selected)
            return f"{hour:02d}:{minute:02d}"

        else:
            self.log("Could not identify two valid hands.")
            return None

    def run(self, image=None, image_name=None):
        # Load the image
        if image is None:
            image_path = self.base / "clocks" / "clock14-10-09.jpg"
            image_name = image_name or image_path.name
        else:
            image_name = image_name or "provided image"

        self.photo_name = image_name

        # Create a separate output folder for each clock
        self.output = self.base / "steps" / Path(image_name).stem
        self.output.mkdir(parents=True, exist_ok=True)

        self.log("Starting photo processing")

        if image is None:
            image = cv2.imread(str(image_path))
            if image is None:
                self.log(f"Cannot open image: {image_path}")
                return f"Cannot open image: {image_path}"

        original_image = image.copy()

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.save('grayscaled_image', gray_image)

        # Gaussian Blur
        blurred_image = cv2.GaussianBlur(
            gray_image, (15, 15), 0
        )

        self.save('blurred_image', blurred_image)

        # Find edges in the image
        edges_display = cv2.Canny(blurred_image, 100, 200)
        self.save('edges_display', edges_display)

        # Detect ellipse candidates
        ellipse_blurred = cv2.GaussianBlur(
            gray_image, (5, 5), 0
        )

        ellipse_edges = cv2.Canny(
            ellipse_blurred, 50, 100
        )

        self.save("ellipse_edges", ellipse_edges)

        h, w = ellipse_edges.shape
        contours, _ = cv2.findContours(
            ellipse_edges.copy(),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE
        )

        ellipses = []

        for contour in contours:

            if len(contour) < 5:
                continue

            ellipse = cv2.fitEllipse(contour)

            (cx, cy), (width, height), ellipse_angle = ellipse

            major_axis = max(width, height)
            minor_axis = min(width, height)

            if minor_axis < min(h, w) * 0.10:
                continue

            if minor_axis / major_axis < 0.4:
                continue

            if major_axis > math.hypot(w, h):
                continue

            if not (0 <= cx < w and 0 <= cy < h):
                continue

            ellipses.append(ellipse)

        ellipses_display = cv2.cvtColor(
            ellipse_edges, cv2.COLOR_GRAY2BGR
        )

        for ellipse in ellipses:
            cv2.ellipse(
                ellipses_display, ellipse, (0, 255, 0), 2
            )

        self.save("ellipses_display", ellipses_display)

        self.log(f"Detected ellipses: {len(ellipses)}")

        # Select the best ellipse
        distance_map = cv2.distanceTransform(
            cv2.bitwise_not(ellipse_edges),
            cv2.DIST_L2,
            5
        )

        best_ellipse = None
        axis_ratio = 1.0

        if ellipses:

            best_ellipse = max(
                ellipses,
                key=lambda e: (
                    0.65 * self.ellipse_score(e, distance_map)
                    + 0.35 * self.concentric_score(e, ellipses)
                )
            )

            selected_ellipse = ellipses_display.copy()

            cv2.ellipse(
                selected_ellipse, best_ellipse, (0, 0, 255), 3
            )

            (cx, cy), (width, height), ellipse_angle = best_ellipse

            cv2.circle(
                selected_ellipse,
                (round(cx), round(cy)),
                5, (0, 0, 255), -1
            )

            self.save("selected_ellipse", selected_ellipse)

            axis_ratio = min(width, height) / max(width, height)

            self.log(
                f"Ellipse score: "
                f"{self.ellipse_score(best_ellipse, distance_map):.3f}"
            )

            self.log(f"Axis ratio: {axis_ratio:.3f}")

            score = self.ellipse_score(
                best_ellipse, distance_map
            )

            if score < 0.50:
                self.log("Ellipse rejected: low confidence.")
                best_ellipse = None
                axis_ratio = 1.0

        else:
            self.log("No valid ellipse detected.")

        # Ellipse geometry diagnostics
        if best_ellipse is not None:

            (ex, ey), (ew, eh), ea = best_ellipse

            self.log(f"Ellipse center: ({ex:.1f}, {ey:.1f})")

            self.log(f"Ellipse dimensions: {ew:.1f} x {eh:.1f}")

            self.log(f"Ellipse radius: {max(ew, eh) / 2:.1f}")

        # Geometric rectification
        if best_ellipse is not None and axis_ratio < 0.95:

            self.log("Geometric rectification required.")

            rectified_image = self.rectify_ellipse(
                image, best_ellipse
            )

            self.save("rectified_image", rectified_image)

            # Update image processing
            image = rectified_image.copy()

            gray_image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            blurred_image = cv2.GaussianBlur(
                gray_image, (15, 15), 0
            )

            edges_display = cv2.Canny(blurred_image, 100, 200)

            self.save(
                "rectified_grayscale", gray_image
            )

            self.save(
                "rectified_blurred", blurred_image
            )

            self.save(
                "rectified_edges", edges_display
            )

        else:

            self.log("No geometric rectification required.")

            self.save("rectified_image", image)

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
                break

        rectangular_clock = False

        # Use ellipse geometry if circle detection failed
        if (
            circles is None
            and best_ellipse is not None
            and score >= 0.60
        ):

            self.log("No clock circle detected.")
            self.log("Using ellipse as fallback.")

            (ex, ey), (ew, eh), ea = best_ellipse

            center = (round(ex), round(ey))
            radius = round(max(ew, eh) / 2)

            circles_display = image.copy()

            cv2.circle(
                circles_display,
                center,
                radius,
                (0, 0, 255), 3
            )

            cv2.circle(
                circles_display,
                center,
                5,
                (0, 0, 255), -1
            )

            self.log(f"Ellipse fallback center: {center}")
            self.log(f"Ellipse fallback radius: {radius}")

            self.save(
                "ellipse_fallback_display",
                circles_display
            )

            self.save(
                "center_display",
                circles_display
            )

        elif circles is None:

            self.log("No clock circle detected.")
            self.log("Trying rectangular clock detection.")

            # Use less aggressive smoothing
            rectangle_blurred = cv2.GaussianBlur(
                gray_image, (3, 3), 0
            )

            edges_display = cv2.Canny(
                rectangle_blurred, 50, 100
            )

            self.save(
                "rectangle_edges",
                edges_display
            )

            rectangle_result = self.detect_rectangle(
                image, edges_display
            )

            if rectangle_result is None:

                self.log("No clock face detected.")
                return self.detect_irregular_clock(original_image)

            center, radius, circles_display = rectangle_result

            rectangular_clock = True

            self.log("Rectangular clock detected.")

            self.save(
                "center_display",
                circles_display
            )

        else:

            circles_display = cv2.cvtColor(
                edges_display, cv2.COLOR_GRAY2BGR
            )

            for x, y, r in np.round(circles[0]).astype(int):

                cv2.circle(
                    circles_display,
                    (int(x), int(y)),
                    int(r),
                    (0, 255, 0), 2
                )

            self.save(
                'circles_display',
                circles_display
            )

            # Calculate distance to the closest edge
            distance_map = cv2.distanceTransform(
                cv2.bitwise_not(edges_display),
                cv2.DIST_L2,
                5
            )

            # Find the best circle
            candidates = np.round(circles[0]).astype(int)

            cx, cy, radius = max(
                candidates,
                key=lambda c: self.circle_score(
                    c[0], c[1], c[2], distance_map
                )
            )

            center = (int(cx), int(cy))

            # Refine the best circle
            cx, cy, radius = self.optimize_circle(
                cx, cy, radius, distance_map
            )

            self.log(f"Circle center: ({cx}, {cy})")
            self.log(f"Circle radius: {radius}")

            if best_ellipse is not None:

                (ex, ey), (ew, eh), ea = best_ellipse

                ellipse_radius = max(ew, eh) / 2

                radius_ratio = radius / ellipse_radius

                center_distance = math.hypot(
                    cx - ex, cy - ey
                )

                self.log(f"Circle/ellipse radius ratio: {radius_ratio:.3f}")

                self.log(f"Circle/ellipse center distance: {center_distance:.1f}")

            # Check consistency between ellipse and circle
            if best_ellipse is not None and axis_ratio < 0.95:

                (ex, ey), (ew, eh), ea = best_ellipse

                ellipse_radius = max(ew, eh) / 2

                radius_ratio = radius / ellipse_radius

                center_distance = math.hypot(
                    cx - ex, cy - ey
                )

                if score >= 0.60 and (
                    radius_ratio < 0.60
                    or center_distance > ellipse_radius * 0.30
                ):

                    self.log("Inconsistent Hough circle detected.")

                    cx = round(ex)
                    cy = round(ey)
                    radius = round(ellipse_radius)

                    self.log("Using rectified ellipse geometry.")
                    self.log(f"Corrected center: ({cx}, {cy})")
                    self.log(f"Corrected radius: {radius}")

            # Update the final clock center
            center = (int(cx), int(cy))

            cv2.circle(
                circles_display,
                center,
                int(radius),
                (0, 0, 255), 3
            )

            cv2.circle(
                circles_display,
                center,
                5,
                (0, 0, 255), -1
            )

            self.save(
                'center_display',
                circles_display
            )

        # Create a circular mask
        mask = np.zeros_like(gray_image)

        cv2.circle(mask, center, int(radius * 0.85), 255, -1)

        self.save('circular_mask', mask)

        # Keep only edges inside the clock
        masked_edges = cv2.bitwise_and(edges_display, edges_display, mask=mask)

        self.save('masked_edges', masked_edges)

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

        self.save('binary_image', binary_image)

        # Check for overlapping hands before selecting two distinct lines
        overlap_result = self.detect_overlapping_hands(binary_image, center, radius)
        if overlap_result is not None:
            self.log(f"Time on the clock: {overlap_result}")
            return overlap_result

        # Detect the potential hands of the clock
        lines_display = image.copy()
        cv2.circle(lines_display, center, radius, (0, 0, 255), 2)
        cv2.circle(lines_display, center, 4, (0, 0, 255), -1)
        simplified_lines_display = image.copy()
        cv2.circle(simplified_lines_display, center, radius, (0, 0, 255), 2)
        cv2.circle(simplified_lines_display, center, 4, (0, 0, 255), -1)

        # Calculate hand thickness
        thickness_map = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

        # Save thickness visualization
        thickness_display = cv2.normalize(
            thickness_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        self.save('thickness_display', thickness_display)

        # Remove thin objects
        kernel_size = max(3, int(radius * 0.035))

        # Ensure an odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        filtered_hands = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        self.save('filtered_hands', filtered_hands)

        # Detect edges of remaining hands
        hands_edges = cv2.Canny(filtered_hands, 90, 180)
        hands_edges = cv2.bitwise_and(hands_edges, hands_edges, mask=mask)

        self.save('hands_edges', hands_edges)

        # Remove thin elements, including the second hand
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        hands_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        self.save('hands_image', hands_image)

        # Detect the remaining edges
        hands_edges = cv2.Canny(hands_image, 90, 180)

        self.save('hands_edges', hands_edges)

        # Initial line detection
        if rectangular_clock:

            lines = cv2.HoughLinesP(
                masked_edges,
                1,
                np.pi / 180,
                threshold=20,
                minLineLength=max(12, int(radius * 0.15)),
                maxLineGap=8
            )

        else:

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

            self.save('detailed_edges', detailed_edges)

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

        self.save('all_lines_display', all_lines_display)

        vectors = []

        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx, dy = self.vector(x1, y1, x2, y2)
                length = self.vector_length(dx, dy)

                # Calculate distance between the clock center and the infinite line

                p = np.array([x1, y1], dtype=float)
                direction = np.array([dx, dy], dtype=float)

                center_point = np.array(center, dtype=float)

                t = np.dot(
                    center_point - p, direction
                ) / max(np.dot(direction, direction), 1)

                closest_point = p + t * direction

                distance = np.linalg.norm(
                    closest_point - center_point
                )

                # Distance between the clock center and the actual segment
                t_segment = np.clip(t, 0.0, 1.0)

                closest_segment_point = p + t_segment * direction

                segment_distance = np.linalg.norm(
                    closest_segment_point - center_point
                )

                if segment_distance > radius * 0.40:
                    continue

                # Reject lines too far from the clock center
                if distance > max(5, radius * 0.30):
                    continue

                cv2.line(
                    lines_display,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                v = [dx, dy, length, x1, y1, x2, y2]

                width = self.hand_width(v, center, thickness_map, radius)

                v.append(width)

                vectors.append(v)

                self.log(
                    f"Angle: {self.angle(v, center):.1f} degrees, "
                    f"Length: {length:.1f}px, Width: {width:.1f}px"
                )

            self.save('lines_display', lines_display)

            # Detect small and long hand

            self.log(f"Detected lines: {len(lines)}")
            self.log(f"Filtered vectors: {len(vectors)}")

            simplified_vectors = self.vector_simplification(vectors, center)

            # Allow close hands before detecting additional lines
            if simplified_vectors is None:

                self.log("Trying a smaller angular threshold.")

                simplified_vectors = self.vector_simplification(
                    vectors,
                    center,
                    angle_threshold=10
                )

            # Retry detection if filtering removed too many lines
            if simplified_vectors is None:

                self.log("Retrying hand detection after filtering.")

                hand_blur = cv2.GaussianBlur(
                    gray_image, (3, 3), 0
                )

                fallback_edges = cv2.Canny(
                    hand_blur, 50, 100
                )

                fallback_edges = cv2.bitwise_and(
                    fallback_edges, fallback_edges, mask=mask
                )

                self.save("fallback_edges", fallback_edges)

                fallback_lines = cv2.HoughLinesP(
                    fallback_edges,
                    1,
                    np.pi / 180,
                    threshold=20,
                    minLineLength=max(12, int(radius * 0.15)),
                    maxLineGap=8
                )

                if fallback_lines is not None:

                    for x1, y1, x2, y2 in fallback_lines.reshape(-1, 4):

                        dx, dy = self.vector(x1, y1, x2, y2)
                        length = self.vector_length(dx, dy)

                        if length == 0:
                            continue

                        distance = abs(
                            dx * (center[1] - y1)
                            - dy * (center[0] - x1)
                        ) / length

                        if distance > max(5, radius * 0.30):
                            continue

                        v = [dx, dy, length, x1, y1, x2, y2]

                        width = self.hand_width(
                            v, center, thickness_map, radius
                        )

                        v.append(width)
                        vectors.append(v)

                    simplified_vectors = self.vector_simplification(
                        vectors, center
                    )

            if simplified_vectors is None:
                self.log("Could not detect two distinct clock hands.")
                if not rectangular_clock and 0 < radius <= 40:
                    small_result = self.detect_small_clock(binary_image, center, radius)
                    if small_result is not None:
                        self.log(f"Time on the clock: {small_result}")
                        return small_result
                return self.detect_irregular_clock(original_image)

            long_hand, short_hand = simplified_vectors

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

            self.save('simplified_lines_display', simplified_lines_display)

            # Angles before pivot correction
            self.log(f"Center before pivot correction: {center}")

            self.log(
                f"Minute angle before pivot correction: "
                f"{self.angle(long_hand, center):.2f}"
            )

            self.log(
                f"Hour angle before pivot correction: "
                f"{self.angle(short_hand, center):.2f}"
            )

            # Calculate angles
            p = np.array(long_hand[3:5], dtype=float)
            q = np.array(short_hand[3:5], dtype=float)
            u = np.array(long_hand[5:7], dtype=float) - p
            v = np.array(short_hand[5:7], dtype=float) - q

            if abs(np.linalg.det(np.column_stack((u, -v)))) > 1e-6:
                t = np.linalg.solve(np.column_stack((u, -v)), q - p)[0]
                pivot = p + t * u

                if np.linalg.norm(pivot - np.array(center)) < radius * 0.35:

                    new_center = tuple(pivot)

                    # Calculate angles before pivot correction
                    old_minute_angle = self.angle(long_hand, center)
                    old_hour_angle = self.angle(short_hand, center)

                    # Calculate angles with the proposed pivot
                    new_minute_angle = self.angle(long_hand, new_center)
                    new_hour_angle = self.angle(short_hand, new_center)

                    # Measure angular variations
                    minute_difference = self.angular_difference(
                        old_minute_angle, new_minute_angle
                    )

                    hour_difference = self.angular_difference(
                        old_hour_angle, new_hour_angle
                    )

                    # Reject pivot corrections that change either angle by more than 90 degrees
                    if minute_difference <= 90 and hour_difference <= 90:

                        center = new_center

                        self.log("Pivot correction accepted.")

                    else:

                        self.log(
                            f"Pivot correction rejected: "
                            f"Minute difference={minute_difference:.2f}, "
                            f"Hour difference={hour_difference:.2f}"
                        )

            self.log(f"Pivot {center}")

            minute_angle = self.angle(long_hand, center)
            hour_angle = self.angle(short_hand, center)

            # Angles after pivot correction
            self.log(
                f"Minute angle after pivot correction: "
                f"{minute_angle:.2f}"
            )

            self.log(
                f"Hour angle after pivot correction: "
                f"{hour_angle:.2f}"
            )

            hour = str(round((hour_angle - minute_angle / 12) / 30) % 12 or 12)

            if len(hour) == 1:
                hour = "0" + hour

            minute = str(round(minute_angle / 6) % 60)

            if len(minute) == 1:
                minute = "0" + minute

            self.log(f"Time on the clock: {hour}:{minute}")
            return hour + ":" + minute

        else:

            self.log("No lines detected.")
            if not rectangular_clock and 0 < radius <= 40:
                small_result = self.detect_small_clock(binary_image, center, radius)
                if small_result is not None:
                    self.log(f"Time on the clock: {small_result}")
                    return small_result
            return self.detect_irregular_clock(original_image)

if __name__ == "__main__":

    base = Path(__file__).resolve().parent

    log_file = base / "logs" / "clock_log.txt"

    clock = ClockVisualization(log_file)

    result = clock.run()

    print("Result:", result)

