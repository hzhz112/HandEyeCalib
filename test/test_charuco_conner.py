# Detect all ChArUco corners.
import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# 1. ChArUco board parameters
# ============================================================

SQUARES_X = 14
SQUARES_Y = 9

SQUARE_LENGTH = 0.020   # Square side: 20 mm
MARKER_LENGTH = 0.015   # ArUco marker side: 15 mm

# D435 color camera configuration
WIDTH = 640
HEIGHT = 480
FPS = 15

WINDOW_NAME = "D435 ChArUco Detection"


# ============================================================
# 2. Create the ChArUco detector
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)


def create_charuco_detector(legacy=False):

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )

    # Compatibility with the legacy ChArUco board layout.
    if legacy:
        board.setLegacyPattern(True)

    detector = cv2.aruco.CharucoDetector(board)

    return board, detector


board, charuco_detector = create_charuco_detector(
    legacy=False
)


# ============================================================
# 3. Normalize the ChArUco corner data
# ============================================================

def normalize_charuco_data(corners, ids):
    """
    Convert corners and IDs into the format OpenCV's drawing functions expect.

    corners: (N, 1, 2), float32
    ids:     (N, 1), int32

    Returns None, None when the data is not usable.
    """

    if corners is None or ids is None:
        return None, None

    corners = np.asarray(corners)
    ids = np.asarray(ids)

    if corners.size == 0 or ids.size == 0:
        return None, None

    # Every corner must carry exactly two coordinates: u and v.
    if corners.size % 2 != 0:
        return None, None

    corners = np.ascontiguousarray(
        corners.reshape(-1, 1, 2),
        dtype=np.float32
    )

    ids = np.ascontiguousarray(
        ids.reshape(-1, 1),
        dtype=np.int32
    )

    # Check that the corner count matches the ID count.
    if corners.shape[0] != ids.shape[0]:
        print(
            "[WARNING] ChArUco data length mismatch:",
            f"corners={corners.shape[0]}",
            f"ids={ids.shape[0]}"
        )

        return None, None

    return corners, ids


# ============================================================
# 4. Set up the RealSense D435
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    WIDTH,
    HEIGHT,
    rs.format.bgr8,
    FPS
)

camera_started = False


try:

    print("Starting RealSense D435...")

    profile = pipeline.start(config)
    camera_started = True

    # Read the color camera intrinsics.
    color_profile = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    intr = color_profile.get_intrinsics()

    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    D = np.array(
        intr.coeffs,
        dtype=np.float64
    )

    print("\n========== Camera info ==========")

    print("Resolution:", intr.width, "x", intr.height)

    print("Camera intrinsics K:")
    print(K)

    print("Distortion coefficients D:")
    print(D)

    print("\n========== Board info ==========")
    print(f"Squares: {SQUARES_X} x {SQUARES_Y}")
    print(f"Square side: {SQUARE_LENGTH * 1000:.1f} mm")
    print(f"Marker side: {MARKER_LENGTH * 1000:.1f} mm")
    print("ArUco dictionary: DICT_5X5_100")

    print("\n========== Controls ==========")
    print("Q: quit")
    print("L: toggle the ChArUco legacy pattern")

    # Create the preview window.
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    legacy_mode = False

    # ========================================================
    # 5. Live detection
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Grab a color frame from the D435
        # ----------------------------------------------------

        try:

            frames = pipeline.wait_for_frames(
                timeout_ms=10000
            )

        except RuntimeError as e:

            print("Failed to grab a frame:", e)
            break

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(
            color_frame.get_data()
        ).copy()

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # ChArUco detection
        # ----------------------------------------------------

        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids
        ) = charuco_detector.detectBoard(gray)

        # Number of ArUco markers.
        marker_count = (
            0 if marker_ids is None
            else len(marker_ids)
        )

        # Normalize the ChArUco corners.
        charuco_corners, charuco_ids = (
            normalize_charuco_data(
                charuco_corners,
                charuco_ids
            )
        )

        # Number of ChArUco corners.
        corner_count = (
            0 if charuco_ids is None
            else len(charuco_ids)
        )

        # ----------------------------------------------------
        # Draw the ArUco markers
        # ----------------------------------------------------

        if marker_ids is not None and marker_count > 0:

            cv2.aruco.drawDetectedMarkers(
                image,
                marker_corners,
                marker_ids
            )

        # ----------------------------------------------------
        # Draw the ChArUco corners
        # ----------------------------------------------------

        if charuco_corners is not None and charuco_ids is not None:

            try:

                cv2.aruco.drawDetectedCornersCharuco(
                    image,
                    charuco_corners,
                    charuco_ids,
                    (0, 255, 255)
                )

            except cv2.error as e:

                print(
                    "\n[WARNING] Failed to draw the ChArUco corners"
                )

                print(
                    "corners shape:",
                    charuco_corners.shape
                )

                print(
                    "ids shape:",
                    charuco_ids.shape
                )

                print("OpenCV error:", e)

        # ----------------------------------------------------
        # Show the detection counts
        # ----------------------------------------------------

        cv2.putText(
            image,
            f"ArUco: {marker_count}/63",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"ChArUco: {corner_count}/104",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        mode_text = (
            "Legacy: ON"
            if legacy_mode
            else "Legacy: OFF"
        )

        cv2.putText(
            image,
            mode_text,
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # Show the live image
        # ----------------------------------------------------

        cv2.imshow(WINDOW_NAME, image)

        # Wait for a key.
        key = cv2.waitKeyEx(10)

        # Debug: show the key code the program actually received.
        if key != -1:
            print(f"Key received: {key}")

        # Q or ESC: quit.
        if key in [ord("q"), ord("Q"), 27]:
            print("Exiting...")
            break

        # L: toggle the legacy pattern.
        elif key in [ord("l"), ord("L")]:
            legacy_mode = not legacy_mode

            board, charuco_detector = create_charuco_detector(
                legacy=legacy_mode
            )

            print(f"Legacy pattern: {legacy_mode}")


finally:

    if camera_started:
        pipeline.stop()

    cv2.destroyAllWindows()

    print("\nD435 closed, program finished.")
