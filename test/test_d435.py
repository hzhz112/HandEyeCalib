
import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream only.
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

try:
    print("Starting D435...")

    profile = pipeline.start(config)

    print("D435 started, waiting for an image...")

    # Warm up and try to grab the first frame.
    for i in range(10):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            color_frame = frames.get_color_frame()

            if color_frame:
                print("First frame acquired.")
                break

        except RuntimeError as e:
            print(f"Attempt {i+1} failed: {e}")

    else:
        raise RuntimeError(
            "No frames after several attempts; check the camera connection"
        )

    print("Opening the preview window...")

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())

        cv2.imshow("D435 RGB", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
