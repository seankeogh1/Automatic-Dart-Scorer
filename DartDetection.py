import cv2
import numpy as np
from BoardDetection import detect_board

# Load the board images before and after adding a dart.
before_image = cv2.imread("Images/After2.jpg", cv2.IMREAD_COLOR)
after_image = cv2.imread("Images/After3.jpg", cv2.IMREAD_COLOR)
# Check that both images loaded.
if before_image is None or after_image is None:
    raise ValueError("Could not load one or both images. Check the file paths.")


# Resize both images using the same dimensions.
height, width = before_image.shape[:2]
scale = min(1600 / width, 1000 / height, 1)
new_size = (int(width * scale), int(height * scale))

before_image = cv2.resize(before_image, new_size)
after_image = cv2.resize(after_image, new_size)


# Find changes between corresponding pixels.
difference = cv2.absdiff(before_image, after_image)
difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Keep changes above the threshold as white pixels.
_, dart_mask = cv2.threshold(
    difference_gray, 20, 255, cv2.THRESH_BINARY
)

# Preserve the mask before closing for comparison.
raw_dart_mask = dart_mask.copy()
# Join small gaps between the barrel and point.
kernel = np.ones((3, 3), dtype=np.uint8)
dart_mask = cv2.morphologyEx(
    dart_mask, cv2.MORPH_CLOSE, kernel
)


# Find outlines of the white shapes in the mask.
contours, _ = cv2.findContours(
    dart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Draw results on a copy, preserving the after image.
dart_preview = after_image.copy()

if contours:
    # Start with the largest changed shape, as in your original code.
    dart_contour = max(contours, key=cv2.contourArea)

    # Create a filled mask containing only this shape.
    component_mask = np.zeros_like(dart_mask)
    cv2.drawContours(
        component_mask, [dart_contour], -1, 255, cv2.FILLED
    )

    # Measure how far each white pixel is from the shape's edge.
    # Thick barrel pixels get larger values than thin cracks.
    thickness = cv2.distanceTransform(
        component_mask, cv2.DIST_L2, 5
    )

    # Prefer thicker pixels when fitting the dart's direction.
    core_y, core_x = np.where(thickness >= 2.0)

    if len(core_x) < 10:
        print("Not enough thick dart pixels to estimate the direction.")

    else:
        core_points = np.column_stack((core_x, core_y))

        # Exclude the upper part, which may contain the flight.
        # Retain the lower 70% of the thick pixels.
        lower_limit = np.percentile(core_points[:, 1], 30)
        fit_points = core_points[
            core_points[:, 1] >= lower_limit
        ].astype(np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            fit_points,
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01
        ).ravel()

        direction = np.array([vx, vy], dtype=np.float32)
        origin = np.array([x0, y0], dtype=np.float32)

        # Make the fitted direction point downwards.
        if direction[1] < 0:
            direction = -direction

        perpendicular = np.array(
            [-direction[1], direction[0]],
            dtype=np.float32
        )

        # Use all boundary pixels for more precise candidate positions.
        detailed_contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        points = max(
            detailed_contours, key=cv2.contourArea
        ).reshape(-1, 2)

        offsets = points.astype(np.float32) - origin

        # Position along the dart and distance from its centreline.
        along = offsets @ direction
        sideways = np.abs(offsets @ perpendicular)

        # Search only near the centreline and beyond the thick core.
        core_along = (fit_points - origin) @ direction
        search_start = np.percentile(core_along, 80)

        candidate_indices = np.flatnonzero(
            (sideways <= 4.0) &
            (along >= search_start)
        )

        # Existing edges may include cracks and board wires.
        # These receive a small penalty, not automatic rejection.
        before_gray = cv2.cvtColor(
            before_image, cv2.COLOR_BGR2GRAY
        )

        existing_edges = cv2.Canny(before_gray, 50, 150)
        existing_edges = cv2.dilate(
            existing_edges,
            np.ones((3, 3), dtype=np.uint8)
        )

       
        evidence_mask = cv2.bitwise_and(
            raw_dart_mask, component_mask
        )

        def sample_support(point, distances):
            """Fraction of sampled positions containing changed pixels."""
            supported = []

            for distance in distances:
                centre = point + distance * direction

                # Search a small strip across the dart centreline.
                samples = (
                    centre[None, :] +
                    np.arange(-1, 2)[:, None] * perpendicular
                )

                coordinates = np.rint(samples).astype(int)
                xs = coordinates[:, 0]
                ys = coordinates[:, 1]

                valid = (
                    (xs >= 0) &
                    (xs < evidence_mask.shape[1]) &
                    (ys >= 0) &
                    (ys < evidence_mask.shape[0])
                )

                if not np.any(valid):
                    return None

                supported.append(
                    np.any(evidence_mask[ys[valid], xs[valid]] > 0)
                )

            return float(np.mean(supported))

        best_point = None
        best_score = -np.inf

        if len(candidate_indices):
            search_length = max(
                float(along[candidate_indices].max() - search_start),
                1.0
            )

            for index in candidate_indices:
                point = points[index].astype(np.float32)
                px, py = points[index]

                # Ignore pixels introduced only by morphological closing.
                if raw_dart_mask[py, px] == 0:
                    continue

                # A tip should have dart pixels leading towards it.
                behind_support = sample_support(
                    point, np.arange(-12, 0)
                )

                # Ideally, the changed shape ends just after the tip.
                ahead_support = sample_support(
                    point, np.arange(2, 9)
                )

                if behind_support is None or ahead_support is None:
                    continue

                # Reject candidates with little evidence behind them.
                if behind_support < 0.50:
                    continue

                alignment = 1.0 - sideways[index] / 4.0
                extension = (
                    along[index] - search_start
                ) / search_length

                existing_edge_penalty = (
                    1.0 if existing_edges[py, px] else 0.0
                )

                # Heuristic score: higher is better.
                score = (
                    2.0 * behind_support +
                    1.0 * (1.0 - ahead_support) +
                    0.7 * alignment +
                    0.8 * extension -
                    0.3 * existing_edge_penalty
                )

                if score > best_score:
                    best_score = score
                    best_point = (int(px), int(py))

        # Show the line fitted through the thicker dart pixels.
        line_start = origin + core_along.min() * direction
        line_end = origin + along.max() * direction

        cv2.line(
            dart_preview,
            tuple(np.rint(line_start).astype(int)),
            tuple(np.rint(line_end).astype(int)),
            (255, 0, 0),
            1
        )

        # Yellow dots show the candidate tip positions.
        for index in candidate_indices:
            px, py = points[index]
            cv2.circle(
                dart_preview, (int(px), int(py)),
                1, (0, 255, 255), -1
            )

        if best_point is not None:
            cv2.circle(
                dart_preview, best_point,
                4, (0, 0, 255), 1
            )

            print("Candidate tip:", *best_point)

        else:
            print("No tip candidate passed the continuity check.")

        cv2.imshow("Thick Dart Core", (thickness >= 2.0).astype(
            np.uint8
        ) * 255)

else:
    print("No changed shapes detected.")



comparison = np.hstack((before_image, after_image))
cv2.imshow("Dart Mask", raw_dart_mask)
cv2.imshow("Dart Detection", dart_preview)

# Keep the windows open until a key is pressed.
cv2.waitKey(0)
cv2.destroyAllWindows()