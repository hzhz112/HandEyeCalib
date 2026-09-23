
import os

import cv2
import numpy as np
import pyrealsense2 as rs

# On Windows, also accept keys typed into a PowerShell / VS Code terminal.
if os.name == "nt":
    import msvcrt


# ============================================================
# 1. Basic parameters
# ============================================================

SQUARES_X = 14
SQUARES_Y = 9

SQUARE_LENGTH = 0.020       # Square side, in metres
MARKER_LENGTH = 0.015       # ArUco marker side, in metres

WIDTH = 640
HEIGHT = 480
FPS = 15

MIN_CORNERS = 12            # Minimum corners needed for pose estimation
AXIS_LENGTH = 0.06          # Axis length, in metres

WINDOW_NAME = "D435 ChArUco Pose"


# ============================================================
# 2. Build the ChArUco board
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

# Legacy OFF has already been verified to detect all 104 corners.
board.setLegacyPattern(False)

charuco_detector = cv2.aruco.CharucoDetector(board)

# 3D position of every ChArUco corner in the board frame.
# Unit: metres.
board_points = np.asarray(
    board.getChessboardCorners(),
    dtype=np.float64
)


# ============================================================
# 3. Keyboard input
# ============================================================

def get_key():
    """
    Accepts keys from either:
    1. the OpenCV image window
    2. a Windows PowerShell / VS Code terminal
    """

    key = cv2.waitKeyEx(10)

    if key != -1:
        return key

    if os.name == "nt" and msvcrt.kbhit():

        ch = msvcrt.getwch()

        return ord(ch.lower())

    return -1


# ============================================================
# 4. Prepare image corners for the RealSense distortion model
# ============================================================

def prepare_pnp_points(image_points, K, intr):
    """
    Input:
        image_points: ChArUco pixel coordinates in the raw image
        K: D435 color camera intrinsics matrix
        intr: RealSense color camera intrinsics

    Output:
        pnp_points: 2D coordinates for solvePnP
        pnp_D: OpenCV distortion coefficients matching those coordinates

    Note:
        inverse_brown_conrady coefficients cannot be used directly as
        OpenCV forward Brown-Conrady distortion coefficients.
    """

    image_points = np.asarray(
        image_points,
        dtype=np.float64
    ).reshape(-1, 2)

    # --------------------------------------------------------
    # Case 1: no distortion
    # --------------------------------------------------------

    if intr.model == rs.distortion.none:

        return (
            image_points,
            np.zeros(5, dtype=np.float64)
        )

    # --------------------------------------------------------
    # Case 2: plain Brown-Conrady
    # --------------------------------------------------------

    if intr.model == rs.distortion.brown_conrady:

        D = np.asarray(
            intr.coeffs,
            dtype=np.float64
        )

        return image_points, D

    # --------------------------------------------------------
    # Case 3: inverse Brown-Conrady
    # --------------------------------------------------------

    if intr.model == rs.distortion.inverse_brown_conrady:

        corrected_points = []

        for u, v in image_points:

            # Turn the raw pixel coordinate into a camera ray.
            #
            # depth=1.0 is used only to obtain the ray direction; it does
            # not represent the actual measured depth of the object.
            x, y, z = rs.rs2_deproject_pixel_to_point(
                intr,
                [float(u), float(v)],
                1.0
            )

            # Normalized ray -> ideal pinhole camera pixel coordinate.
            u_corrected = (
                K[0, 0] * (x / z) + K[0, 2]
            )

            v_corrected = (
                K[1, 1] * (y / z) + K[1, 2]
            )

            corrected_points.append([
                u_corrected,
                v_corrected
            ])

        corrected_points = np.asarray(
            corrected_points,
            dtype=np.float64
        )

        # Already de-distorted, so PnP must not apply distortion again.
        D_zero = np.zeros(5, dtype=np.float64)

        return corrected_points, D_zero

    raise RuntimeError(
        f"Unhandled RealSense distortion model: {intr.model}"
    )


# ============================================================
# 5. ChArUco pose estimation
# ============================================================

def estimate_board_pose(
    charuco_corners,
    charuco_ids,
    K,
    intr
):
    """
    Compute the transform from the board frame to the D435 color camera frame:

        ^C T_B

    Returns:
        rvec
        tvec
        R
        T
        rms
        corner_count

    Returns None when there is not enough usable detection data.
    """

    # --------------------------------------------------------
    # Validate the detection
    # --------------------------------------------------------

    if charuco_corners is None or charuco_ids is None:
        return None

    image_points = np.asarray(
        charuco_corners,
        dtype=np.float64
    ).reshape(-1, 2)

    ids = np.asarray(
        charuco_ids,
        dtype=np.int32
    ).reshape(-1)

    if len(image_points) != len(ids):
        return None

    if len(ids) < MIN_CORNERS:
        return None

    if np.any(ids < 0):
        return None

    if np.any(ids >= len(board_points)):
        return None

    if len(np.unique(ids)) != len(ids):
        return None

    # --------------------------------------------------------
    # Look up the board-frame 3D position of every corner
    # --------------------------------------------------------

    object_points = np.ascontiguousarray(
        board_points[ids],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Prepare the 2D corners for the camera distortion model
    # --------------------------------------------------------

    pnp_points, pnp_D = prepare_pnp_points(
        image_points,
        K,
        intr
    )

    pnp_points = np.ascontiguousarray(
        pnp_points,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Estimate the pose with PnP
    # --------------------------------------------------------

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        pnp_points,
        K,
        pnp_D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # The board must be in front of the camera.
    if float(tvec[2, 0]) <= 0:
        return None

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R, _ = cv2.Rodrigues(rvec)

    # --------------------------------------------------------
    # 4x4 homogeneous transform
    # --------------------------------------------------------

    T = np.eye(4, dtype=np.float64)

    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    # --------------------------------------------------------
    # Reprojection error
    # --------------------------------------------------------

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        pnp_D
    )

    projected_points = projected_points.reshape(-1, 2)

    errors = np.linalg.norm(
        projected_points - pnp_points,
        axis=1
    )

    rms = float(
        np.sqrt(np.mean(errors ** 2))
    )

    return {
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "T": T,
        "rms": rms,
        "corner_count": len(ids)
    }


# ============================================================
# 6. Project 3D axes onto the raw camera image
# ============================================================

def project_points_to_raw_image(
    object_points,
    pose,
    K,
    intr
):
    """
    Project board-frame 3D points onto the raw D435 color image.

    For inverse_brown_conrady:
        first project into the ideal pinhole image, then apply the inverse
        distortion model to get the matching raw-image pixel coordinates.
    """

    object_points = np.asarray(
        object_points,
        dtype=np.float64
    ).reshape(-1, 3)

    rvec = pose["rvec"]
    tvec = pose["tvec"]

    # --------------------------------------------------------
    # No distortion
    # --------------------------------------------------------

    if intr.model == rs.distortion.none:

        pixels, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            np.zeros(5, dtype=np.float64)
        )

        return pixels.reshape(-1, 2)

    # --------------------------------------------------------
    # Plain Brown-Conrady
    # --------------------------------------------------------

    if intr.model == rs.distortion.brown_conrady:

        pixels, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            np.asarray(intr.coeffs, dtype=np.float64)
        )

        return pixels.reshape(-1, 2)

    # --------------------------------------------------------
    # Inverse Brown-Conrady
    # --------------------------------------------------------

    if intr.model == rs.distortion.inverse_brown_conrady:

        # Step 1: project into the ideal pinhole image.
        ideal_pixels, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            np.zeros(5, dtype=np.float64)
        )

        ideal_pixels = np.ascontiguousarray(
            ideal_pixels,
            dtype=np.float64
        )

        # Step 2:
        # inverse_brown_conrady coefficients describe
        # raw normalized pixel coordinates -> ideal normalized coordinates.
        #
        # OpenCV's inverse solve maps the ideal points back to their
        # positions in the raw image.
        raw_pixels = cv2.undistortPoints(
            ideal_pixels,
            K,
            np.asarray(intr.coeffs, dtype=np.float64),
            P=K
        )

        return raw_pixels.reshape(-1, 2)

    raise RuntimeError(
        f"Unhandled distortion model: {intr.model}"
    )


def draw_pose_axes(image, pose, K, intr):
    """
    Draw the board frame axes:
        X: red
        Y: green
        Z: blue
    """

    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [AXIS_LENGTH, 0.0, 0.0],
        [0.0, AXIS_LENGTH, 0.0],
        [0.0, 0.0, AXIS_LENGTH]
    ], dtype=np.float64)

    pixels = project_points_to_raw_image(
        axis_points,
        pose,
        K,
        intr
    )

    if not np.isfinite(pixels).all():
        return

    origin, x_point, y_point, z_point = (
        np.round(pixels).astype(int)
    )

    origin = tuple(origin)
    x_point = tuple(x_point)
    y_point = tuple(y_point)
    z_point = tuple(z_point)

    cv2.line(
        image, origin, x_point,
        (0, 0, 255), 3
    )

    cv2.line(
        image, origin, y_point,
        (0, 255, 0), 3
    )

    cv2.line(
        image, origin, z_point,
        (255, 0, 0), 3
    )

    cv2.circle(
        image,
        origin,
        4,
        (255, 255, 255),
        -1
    )

    cv2.putText(
        image, "X", x_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 255), 2
    )

    cv2.putText(
        image, "Y", y_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2
    )

    cv2.putText(
        image, "Z", z_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 0, 0), 2
    )


# ============================================================
# 7. Set up the D435 camera
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream only.
config.enable_stream(
    rs.stream.color,
    WIDTH,
    HEIGHT,
    rs.format.bgr8,
    FPS
)

camera_started = False


try:

    print("Starting D435...")

    profile = pipeline.start(config)
    camera_started = True

    # --------------------------------------------------------
    # Read the color camera intrinsics
    # --------------------------------------------------------

    color_profile = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    intr = color_profile.get_intrinsics()

    K = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    print("\n========== D435 camera info ==========")

    print("Resolution:", intr.width, "x", intr.height)

    print("Camera intrinsics K:")
    print(K)

    print("\nDistortion model:")
    print(intr.model)

    print("\nRealSense distortion coefficients:")
    print(np.asarray(intr.coeffs))

    # Check whether this model is supported by the program.
    supported_models = (
        rs.distortion.none,
        rs.distortion.brown_conrady,
        rs.distortion.inverse_brown_conrady
    )

    if intr.model not in supported_models:
        raise RuntimeError(
            f"This program does not support distortion model: {intr.model}"
        )

    if intr.model == rs.distortion.inverse_brown_conrady:
        print(
            "\ninverse_brown_conrady corner de-distortion is enabled."
        )

    print("\n========== Board info ==========")

    print("Squares: 14 x 9")
    print("Square side: 20 mm")
    print("Marker side: 15 mm")
    print("Dictionary: DICT_5X5_100")
    print("Legacy: OFF")
    print("Board interior corners:", len(board_points))

    print("\n========== Controls ==========")

    print("P: print the current board pose")
    print("Q / ESC: quit")
    print("Keys work in either the OpenCV window or a Windows terminal")
    print()

    # --------------------------------------------------------
    # Create the window
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # Raw 640x480 image plus a separate 120 px info panel below it.
    cv2.resizeWindow(
        WINDOW_NAME,
        WIDTH,
        HEIGHT + 120
    )

    latest_pose = None

    # ========================================================
    # 8. Live detection and pose estimation
    # ========================================================

    while True:

        color_frame = None

        # ----------------------------------------------------
        # Grab a color frame
        # ----------------------------------------------------

        try:

            frames = pipeline.wait_for_frames(
                timeout_ms=1000
            )

            color_frame = frames.get_color_frame()

        except RuntimeError:
            pass

        if color_frame:

            image = np.asanyarray(
                color_frame.get_data()
            ).copy()

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # ------------------------------------------------
            # Detect ChArUco
            # ------------------------------------------------

            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids
            ) = charuco_detector.detectBoard(gray)

            marker_count = (
                0 if marker_ids is None
                else len(marker_ids)
            )

            corner_count = (
                0 if charuco_ids is None
                else len(charuco_ids)
            )

            # ------------------------------------------------
            # Draw the detection results
            # ------------------------------------------------

            if (
                marker_ids is not None
                and marker_count > 0
                and len(marker_corners) == marker_count
            ):

                cv2.aruco.drawDetectedMarkers(
                    image,
                    marker_corners,
                    marker_ids
                )

            if charuco_corners is not None and charuco_ids is not None:

                # Convert to NumPy arrays.
                points = np.asarray(charuco_corners)
                ids_array = np.asarray(charuco_ids)

                # Verify that every corner carries two pixel coordinates.
                if points.size % 2 == 0:

                    # Normalize to an N x 2 array of 2D coordinates.
                    points = points.reshape(-1, 2)

                    # Flatten the IDs to a 1D array.
                    ids_array = ids_array.reshape(-1)

                    # Corners and IDs must correspond one to one.
                    if len(points) == len(ids_array):

                        for point in points:

                            u = int(round(float(point[0])))
                            v = int(round(float(point[1])))

                            # Yellow dot marking a ChArUco corner.
                            cv2.circle(
                                image,
                                (u, v),
                                3,
                                (0, 255, 255),
                                -1
                            )

                    else:

                        print(
                            "[WARNING] ChArUco data length mismatch:",
                            "points =", len(points),
                            "ids =", len(ids_array)
                        )

            # ------------------------------------------------
            # Estimate the board pose
            # ------------------------------------------------

            pose = estimate_board_pose(
                charuco_corners,
                charuco_ids,
                K,
                intr
            )

            # Updated every frame so that P never prints stale data.
            latest_pose = pose

            # ------------------------------------------------
            # Draw the 3D axes
            # ------------------------------------------------

            if pose is not None:

                draw_pose_axes(
                    image,
                    pose,
                    K,
                    intr
                )

            # ------------------------------------------------
            # Separate info panel below, so the raw image stays clear
            # ------------------------------------------------

            panel = np.zeros(
                (120, image.shape[1], 3),
                dtype=np.uint8
            )

            cv2.putText(
                panel,
                f"ArUco: {marker_count}/63"
                f"    ChArUco: {corner_count}/104",
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            if pose is not None:

                x, y, z = pose["tvec"].reshape(3)

                cv2.putText(
                    panel,
                    f"X: {x * 1000:+.1f} mm   "
                    f"Y: {y * 1000:+.1f} mm   "
                    f"Z: {z * 1000:+.1f} mm",
                    (10, 57),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.53,
                    (0, 255, 0),
                    1
                )

                cv2.putText(
                    panel,
                    f"Reprojection RMS: {pose['rms']:.3f} px",
                    (10, 87),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1
                )

            else:

                cv2.putText(
                    panel,
                    "Pose unavailable: insufficient valid corners",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1
                )

            cv2.putText(
                panel,
                "P: Print pose     Q / ESC: Quit",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (180, 180, 180),
                1
            )

            # Raw image plus info panel.
            display = np.vstack([
                image,
                panel
            ])

            cv2.imshow(
                WINDOW_NAME,
                display
            )

        # ----------------------------------------------------
        # Handle keyboard input
        # ----------------------------------------------------

        key = get_key()

        # Q / ESC: quit.
        if key in (ord("q"), ord("Q"), 27):

            print("\nExiting...")
            break

        # P: print the current pose.
        elif key in (ord("p"), ord("P")):

            if latest_pose is None:

                print(
                    "\nNo valid pose right now;"
                    " make sure the board is clearly visible."
                )

            else:

                print(
                    "\n========== Board pose relative to the camera =========="
                )

                print(
                    "Valid corner count:",
                    latest_pose["corner_count"]
                )

                print("\nRotation vector rvec:")
                print(latest_pose["rvec"])

                print("\nTranslation vector tvec (metres):")
                print(latest_pose["tvec"])

                print("\nRotation matrix R:")
                print(latest_pose["R"])

                print("\nHomogeneous transform ^C T_B:")
                print(latest_pose["T"])

                print(
                    "\nReprojection RMS:",
                    f"{latest_pose['rms']:.4f} px"
                )

                print(
                    "=======================================================\n"
                )

        # ----------------------------------------------------
        # Closing the window with its X button also quits.
        # ----------------------------------------------------

        try:

            visible = cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            )

            if visible < 1:
                print("\nWindow closed, exiting...")
                break

        except cv2.error:
            break


except KeyboardInterrupt:

    print("\nCtrl+C received, exiting...")


finally:

    if camera_started:
        pipeline.stop()

    cv2.destroyAllWindows()

    print("D435 closed, program finished.")
