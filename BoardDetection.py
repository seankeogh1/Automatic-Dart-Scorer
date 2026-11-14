import cv2
import numpy as np


def detect_board(image_path, show_debug=False):
    # Load and resize the board image.
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]
    scale = min(800 / width, 700 / height)
    img = cv2.resize(img, (int(width * scale), int(height * scale)))

    clean_image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect red and green areas.
    green_mask = cv2.inRange(hsv, (35, 80, 50), (90, 255, 255))

    red_mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 80, 50), (179, 255, 255))

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    colour_mask = cv2.bitwise_or(green_mask, red_mask)

    # Close wire gaps and find the main ring boundaries.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    board_mask = cv2.morphologyEx(
        colour_mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        board_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    sorted_contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )

    if len(sorted_contours) < 5:
        raise ValueError("Not enough board contours detected.")

    outer_double, inner_double, outer_treble, inner_treble = (
        sorted_contours[:4]
    )

    if show_debug:
        for index, contour in enumerate(sorted_contours[:4]):
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                colour = (
                    (255, 0, 0) if index % 2 == 0
                    else (0, 255, 255)
                )
                cv2.ellipse(img, ellipse, colour, 2)

    # Find the outer bull inside the inner treble boundary.
    outer_bull_candidates = []

    for contour in sorted_contours[4:]:
        inside = all(
            cv2.pointPolygonTest(
                inner_treble,
                (float(point[0][0]), float(point[0][1])),
                False
            ) > 0
            for point in contour
        )

        if inside:
            outer_bull_candidates.append(contour)

    if not outer_bull_candidates:
        raise ValueError("Outer bull not found.")

    outer_bull = max(outer_bull_candidates, key=cv2.contourArea)

    # Find the red bullseye and use its centre as the board centre.
    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    bull_candidates = []

    for contour in red_contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)

            if cv2.pointPolygonTest(
                outer_bull, ellipse[0], False
            ) >= 0:
                bull_candidates.append(contour)

    if not bull_candidates:
        raise ValueError("Inner bull not found.")

    inner_bull = max(bull_candidates, key=cv2.contourArea)
    inner_bull_ellipse = cv2.fitEllipse(inner_bull)
    centre_x, centre_y = inner_bull_ellipse[0]

    if show_debug:
        cv2.drawContours(img, [outer_bull], -1, (255, 0, 0), 2)
        cv2.ellipse(img, inner_bull_ellipse, (0, 255, 255), 2)
        cv2.circle(
            img, (int(centre_x), int(centre_y)),
            1, (255, 255, 255), -1
        )

    # Helpers for creating filled areas and rings.
    def filled_mask(contour):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        return mask

    def ring_mask(outside_contour, inside_contour):
        mask = filled_mask(outside_contour)
        cv2.drawContours(
            mask, [inside_contour], -1, 0, cv2.FILLED
        )
        return mask

    # Create scoring masks for doubles, trebles and singles.
    double_mask = ring_mask(outer_double, inner_double)
    treble_mask = ring_mask(outer_treble, inner_treble)
    outer_single_mask = ring_mask(inner_double, outer_treble)
    inner_single_mask = ring_mask(inner_treble, outer_bull)

    single_mask = cv2.bitwise_or(
        outer_single_mask, inner_single_mask
    )

    # Separate the black and white single sections.
    section_kernel = np.ones((3, 3), dtype=np.uint8)
    singles_mask = cv2.erode(
        single_mask, section_kernel, iterations=1
    )

    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    white_mask = cv2.inRange(blurred_gray, 155, 255)
    black_mask = cv2.inRange(blurred_gray, 0, 105)

    white_mask = cv2.bitwise_and(white_mask, singles_mask)
    black_mask = cv2.bitwise_and(black_mask, singles_mask)

    white_mask = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, section_kernel
    )
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_OPEN, section_kernel
    )

    white_contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    black_contours, _ = cv2.findContours(
        black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_section_area = cv2.contourArea(outer_double) * 0.001

    white_sections = [
        contour for contour in white_contours
        if cv2.contourArea(contour) > min_section_area
    ]
    black_sections = [
        contour for contour in black_contours
        if cv2.contourArea(contour) > min_section_area
    ]

    if show_debug:
        cv2.drawContours(img, white_sections, -1, (255, 255, 0), 1)
        cv2.drawContours(img, black_sections, -1, (255, 0, 255), 1)

    # Use the outer single sections to identify wedge directions.
    outer_sections = []

    for section in white_sections + black_sections:
        moments = cv2.moments(section)

        if moments["m00"] == 0:
            continue

        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]

        if cv2.pointPolygonTest(outer_treble, (x, y), False) < 0:
            outer_sections.append(section)

    def section_angle(section):
        moments = cv2.moments(section)
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        return np.arctan2(y - centre_y, x - centre_x)

    outer_sections = sorted(outer_sections, key=section_angle)

    # Build a wedge from a section's side angles, excluding the bull.
    board_limit = ring_mask(outer_double, outer_bull)
    pixel_y, pixel_x = np.indices(gray.shape)

    def build_wedge(section):
        points = section.reshape(-1, 2)
        reference_angle = section_angle(section)

        angles = np.arctan2(
            points[:, 1] - centre_y,
            points[:, 0] - centre_x
        )
        angles = (
            angles - reference_angle + np.pi
        ) % (2 * np.pi) - np.pi

        pixel_angles = np.arctan2(
            pixel_y - centre_y,
            pixel_x - centre_x
        )
        pixel_angles = (
            pixel_angles - reference_angle + np.pi
        ) % (2 * np.pi) - np.pi

        wedge_mask = (
            (pixel_angles >= angles.min()) &
            (pixel_angles <= angles.max())
        ).astype(np.uint8) * 255

        return cv2.bitwise_and(wedge_mask, board_limit)

    board_numbers = [
        20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
        3, 19, 7, 16, 8, 11, 14, 9, 12, 5
    ]

    selection_background = img.copy()
    scoring_areas = []
    twenty_selected = False
    testing = False
    test_click = None

    # Select 20 for calibration, then score test clicks.
    def twenty_click(event, x, y, flags, param):
        nonlocal twenty_selected, testing, test_click

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
            return

        on_button = 10 <= x <= 175 and 10 <= y <= 50

        if testing:
            if on_button:
                return

            for area in scoring_areas:
                if area["mask"][y, x] != 0:
                    test_click = area["score"]
                    print(f'{area["label"]}: {test_click}')
                    return

            test_click = 0
            print("Miss or unassigned area: 0")
            return

        if on_button:
            if twenty_selected:
                testing = True
                print("20 confirmed. Click the board to test scores.")
            return

        for section_index, section in enumerate(outer_sections):
            if cv2.pointPolygonTest(section, (x, y), False) < 0:
                continue

            # Number the detected sections clockwise from the selected 20.
            if len(outer_sections) != 20:
                print(
                    f"Detected {len(outer_sections)} outer sections; "
                    "20 are needed for correct numbering."
                )
                return

            twenty_selected = True
            ordered_sections = (
                outer_sections[section_index:] +
                outer_sections[:section_index]
            )

            display = selection_background.copy()
            scoring_areas.clear()

            for number, scoring_section in zip(
                board_numbers, ordered_sections
            ):
                numbered_wedge = build_wedge(scoring_section)

                for region_mask, score, label in [
                    (single_mask, number, f"Single {number}"),
                    (double_mask, number * 2, f"Double {number}"),
                    (treble_mask, number * 3, f"Treble {number}")
                ]:
                    scoring_areas.append({
                        "mask": cv2.bitwise_and(
                            numbered_wedge, region_mask
                        ),
                        "score": score,
                        "label": label
                    })

                # Position each number just outside the double ring.
                label_angle = section_angle(scoring_section)
                boundary_points = outer_double.reshape(-1, 2)

                boundary_angles = np.arctan2(
                    boundary_points[:, 1] - centre_y,
                    boundary_points[:, 0] - centre_x
                )
                angle_difference = np.abs(
                    (boundary_angles - label_angle + np.pi)
                    % (2 * np.pi) - np.pi
                )

                boundary_x, boundary_y = boundary_points[
                    np.argmin(angle_difference)
                ]

                label_x = int(boundary_x + 18 * np.cos(label_angle))
                label_y = int(boundary_y + 18 * np.sin(label_angle))

                cv2.putText(
                    display, str(number), (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA
                )

            # Add the outer bull and bullseye scoring areas.
            scoring_areas.extend([
                {
                    "mask": ring_mask(outer_bull, inner_bull),
                    "score": 25,
                    "label": "Outer bull"
                },
                {
                    "mask": filled_mask(inner_bull),
                    "score": 50,
                    "label": "Bullseye"
                }
            ])

            # Highlight 20 and display the Start Test button.
            wedge_contours, _ = cv2.findContours(
                build_wedge(section),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                display, wedge_contours, -1, (0, 255, 255), 2
            )

            cv2.rectangle(
                display, (10, 10), (175, 50),
                (50, 120, 50), cv2.FILLED
            )
            cv2.putText(
                display, "Start Test", (20, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA
            )

            print(f"Saved {len(scoring_areas)} scoring areas.")
            cv2.imshow("Board Image", display)
            break

    # Open the interactive calibration and testing window.
    if show_debug:
        cv2.imshow("Board Image", img)
        cv2.setMouseCallback("Board Image", twenty_click)
        cv2.moveWindow("Board Image", 100, 100)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return detected board information and the last test-click score.
    return {
        "image": clean_image,
        "outer_double": outer_double,
        "inner_double": inner_double,
        "outer_treble": outer_treble,
        "inner_treble": inner_treble,
        "bull_centre": (centre_x, centre_y),
        "outer_sections": outer_sections,
        "test_click": test_click
    }


if __name__ == "__main__":
    board = detect_board(
        "Images/EmptyBoardTest.jpg",
        show_debug=True
    )

    print("Returned test click:", board["test_click"])