import cv2

from BoardDetection import detect_board, get_score
from DartDetection import detect_dart


def load_images(before_path, after_path):
    before = cv2.imread(before_path, cv2.IMREAD_COLOR)
    after = cv2.imread(after_path, cv2.IMREAD_COLOR)

    if before is None or after is None:
        raise ValueError("Could not load images. Check the file paths.")

    if before.shape != after.shape:
        raise ValueError(
            "Capture both images at the same camera resolution."
        )

    # Preserve the image size used by your board detection.
    height, width = before.shape[:2]
    scale = min(800 / width, 700 / height)
    size = (int(width * scale), int(height * scale))

    before = cv2.resize(before, size)
    after = cv2.resize(after, size)

    return before, after


def score_throw(before_path, after_path):
    before, after = load_images(before_path, after_path)

    # Select the outer single 20, click Start Test, then press a key.
    print("Select the outer single 20 section.")
    print("Click Start Test, then press a key to continue.")

    # Calibrate using an unobstructed board image.
    calibration_image = cv2.imread(
        "Images/Before.jpg",
        cv2.IMREAD_COLOR
    )

    if calibration_image is None:
        raise ValueError("Could not load the calibration image.")

    height, width = before.shape[:2]
    calibration_image = cv2.resize(
        calibration_image, (width, height)
    )

    board = detect_board(calibration_image, show_debug=True)

    if not board["calibrated"]:
        raise ValueError(
            "Calibration was not completed. "
            "Select 20 and click Start Test before pressing a key."
        )

    # Find the new dart in the same coordinate system as the board.
    tip = detect_dart(before, after, show_debug=False)

    if tip is None:
        print("No dart tip detected.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    x, y = tip
    score = get_score(board, x, y)

    # Overlay scoring regions on the after image.
    preview = after.copy()

    for area in board["scoring_areas"]:
        contours, _ = cv2.findContours(
            area["mask"],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            preview, contours, -1, (0, 255, 255), 1
        )

    # Mark the detected landing position.
    cv2.circle(preview, (x, y), 5, (0, 0, 255), 2)

    cv2.putText(
        preview,
        f"Score: {score}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    print(f"Dart tip: ({x}, {y})")
    print(f"Score: {score}")

    cv2.imshow("Scoring Result", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return score


if __name__ == "__main__":
    result = score_throw(
        "Images/After.jpg",
        "Images/After2.jpg"
    )

    print("Returned score:", result)