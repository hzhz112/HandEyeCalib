# ArUco marker detection.
import cv2
import numpy as np
import pyrealsense2 as rs

# Set up the RealSense D435.
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

# Start the camera.
profile = pipeline.start(config)

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

print("Camera intrinsics K:")
print(K)

# Start with one candidate dictionary.
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)

detector = cv2.aruco.ArucoDetector(
    dictionary,
    cv2.aruco.DetectorParameters()
)

# Build the ChArUco board model.
board = cv2.aruco.CharucoBoard(
    (14, 9),
    0.020,
    0.015,
    dictionary
)

# Create the ChArUco detector.
charuco_detector = cv2.aruco.CharucoDetector(board)

print("D435 started, press q to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(
            color_frame.get_data()
        ).copy()

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        # Detect the ArUco markers inside the board.
        corners, ids, rejected = detector.detectMarkers(
            gray
        )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                image, corners, ids
            )

            cv2.putText(
                image,
                f"Markers: {len(ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("D435 ChArUco Test", image)

        key = cv2.waitKeyEx(10)
        if key != -1:
            print(f"Key received: {key}")

        # Q or ESC: quit.
        if key in [ord("q"), ord("Q"), 27]:
            print("Exiting...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
