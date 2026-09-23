"""Interactive ChArUco handeye data collection.

Flow control only. This module wires the configured adapters together, pumps
the camera and the OpenCV window, and delegates detection to `board/` and
persistence to `dataset/`. It knows nothing about the RealSense or RealMan
SDKs, nothing about ChArUco internals, and nothing about the JSON schema.

Run from the repository root:

    python -m calibration.apps.collect_handeye [--config PATH]

`python calibration/apps/collect_handeye.py` does not work: that puts the
script's own directory on sys.path[0] and the `calibration.*` imports fail.

Press S to save one sample, Q or ESC to quit. The arm must be stationary.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.board.charuco_detector import (
    BoardDetection,
    CharucoBoardSpec,
    CharucoDetector,
)
from calibration.camera.camera_factory import create_camera
from calibration.config.loader import DEFAULT_CONFIG_PATH, HandEyeConfig, load_config
from calibration.dataset.sample import HandEyeSample
from calibration.dataset.writer import DatasetWriter
from calibration.robot.robot_factory import create_robot
from calibration.robot.robot_interface import validate_transform_matrix

if os.name == "nt":
    import msvcrt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect ChArUco handeye calibration samples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def get_key() -> int:
    """Return a pressed key code, or -1.

    waitKeyEx also pumps the window, so it must be called on every iteration.
    The msvcrt fallback catches keys the window never sees, and lowercases.
    """
    key = cv2.waitKeyEx(10)
    if key != -1:
        return key
    if os.name == "nt" and msvcrt.kbhit():
        return ord(msvcrt.getwch().lower())
    return -1


def build_panel(
    width: int,
    detection: BoardDetection | None,
    saved: int,
    detector: CharucoDetector,
    max_rms: float,
    panel_height: int,
) -> np.ndarray:
    """Render the status strip shown below the image.

    `detection is None` means no frame arrived, which is a different state from
    a frame that arrived with no board in it.
    """
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)

    if detection is None:
        cv2.putText(panel, "NO SIGNAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(panel, f"Saved:{saved}   (waiting for camera)", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return panel

    text = (f"ArUco: {detection.marker_count}/{detector.marker_total}   "
            f"ChArUco: {detection.corner_count}/{detector.corner_total}")
    cv2.putText(panel, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if detection.T_camera_board is not None:
        x, y, z = detection.T_camera_board[:3, 3] * 1000
        text = f"X:{x:+.1f}  Y:{y:+.1f}  Z:{z:+.1f} mm"
        cv2.putText(panel, text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        status = "READY" if detection.rms <= max_rms else "RMS TOO HIGH"
        text = f"RMS:{detection.rms:.3f}px   {status}   Saved:{saved}"
    else:
        text = f"Pose unavailable   Saved:{saved}"

    cv2.putText(panel, text, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return panel


def capture_sample(robot, latest, writer: DatasetWriter, config: HandEyeConfig) -> None:
    """Read the robot and hand one paired sample to the writer.

    Every gate is checked before the robot is read, so a rejected frame costs
    no robot I/O. The arm must be stationary: the two devices are not
    hardware-synchronized.
    """
    if latest is None:
        print("[SKIP] No camera frame received yet")
        return

    # Unpacked once, at the top: every additional unpacking site is another
    # chance to miss one and have the failure swallowed by the caller's broad
    # except, silently discarding samples.
    frame = latest["frame"]
    detection = latest["detection"]

    # Only the monotonic clock is safe here. host_time is wall clock and can
    # jump under NTP, which would make a fresh frame look stale or vice versa.
    age = time.monotonic() - frame.monotonic_time
    if age > config.collector.max_frame_age:
        print(f"[SKIP] Frame is stale: {age:.2f} s")
        return

    if detection.T_camera_board is None:
        print("[SKIP] No valid board pose in the current frame")
        return
    if detection.rms > config.collector.max_rms:
        print(f"[SKIP] Reprojection error too high: {detection.rms:.3f} px")
        return

    # The midpoint of the read, not a single sample, because reading the arm
    # over TCP takes a non-trivial and variable amount of time.
    robot_read_start = time.time()
    T_base_flange = validate_transform_matrix(robot.get_base_T_flange())
    robot_read_end = time.time()
    robot_time = (robot_read_start + robot_read_end) / 2.0

    sample = HandEyeSample(
        index=0,  # assigned by the writer, from its committed count
        camera_time=frame.host_time,
        robot_time=robot_time,
        device_timestamp_ms=frame.device_timestamp_ms,
        T_base_flange=T_base_flange,
        T_camera_board=detection.T_camera_board,
        image_path="",  # assigned by the writer
        reprojection_error=detection.rms,
        aruco_markers=detection.marker_count,
        charuco_corners=detection.corner_count,
    )

    sample = writer.save(sample, frame.image)

    print(f"[SAVED] sample {sample.index} | corners: {sample.charuco_corners} | "
          f"rms: {sample.reprojection_error:.3f} px")
    print(f"T_base_flange (m):\n{sample.T_base_flange}")
    print(f"Data file: {writer.json_path}\n")


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)

    detector = CharucoDetector(
        CharucoBoardSpec(
            squares_x=config.board.squares_x,
            squares_y=config.board.squares_y,
            square_length=config.board.square_length,
            marker_length=config.board.marker_length,
            dictionary=config.board.dictionary,
            legacy_pattern=config.board.legacy_pattern,
        ),
        min_corners=config.collector.min_corners,
    )

    robot = create_robot(config.robot)
    camera = create_camera(config.camera)

    # Created before any device is opened, so an aborted run still leaves a
    # session directory behind.
    writer = DatasetWriter(config.output_root, config.board, camera, robot)

    robot_connected = camera_started = False
    latest = None
    last_display = None
    last_width = None

    try:
        print(f"Connecting to robot: {robot.name} ({config.robot.type})")
        robot.connect()
        robot_connected = True
        print(f"{robot.name} connected. Current T_base_flange (m):\n"
              f"{validate_transform_matrix(robot.get_base_T_flange())}")

        camera.start()
        camera_started = True

        intrinsics = camera.get_intrinsics()
        # Seed the panel width, so the very first camera timeout still has a
        # width to draw the NO SIGNAL panel with.
        last_width = intrinsics.width

        print(f"{camera.name} started | distortion: {intrinsics.distortion_model} | "
              f"{intrinsics.width}x{intrinsics.height}@{intrinsics.fps}")
        print(f"Board: {detector.corner_total} corners / {detector.marker_total} markers")
        print(f"Config: {config.config_path}")
        print("S: save sample | Q / ESC: quit")
        print("Keep the arm stationary before pressing S")
        print("Session directory:", writer.session_dir)

        cv2.namedWindow(config.collector.window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            # A timeout returns None and is not an error: keep polling the
            # keyboard so the window stays responsive instead of feeling hung.
            frame = camera.get_frame(timeout_ms=config.collector.frame_timeout_ms)

            if frame is not None:
                detection = detector.detect(frame.image, camera)
                # Frame and detection stay paired: the staleness gate reads the
                # timestamp of the very frame this detection came from.
                latest = {"frame": frame, "detection": detection}

                last_width = frame.image.shape[1]
                # Copy before drawing: adapters never hand out a buffer the
                # caller is expected to mutate in place.
                last_display = frame.image.copy()
                # The raw corners, not the de-distorted ones: under
                # inverse_brown_conrady those are different coordinates.
                for u, v in detection.points:
                    cv2.circle(last_display, (int(round(u)), int(round(v))), 2, (0, 255, 255), -1)

                panel = build_panel(
                    last_width, detection, writer.count, detector,
                    config.collector.max_rms, config.collector.panel_height,
                )
            else:
                # Redraw even without a frame, otherwise the window freezes on
                # the last image and looks like the program has died.
                if last_display is None:
                    last_display = np.zeros((intrinsics.height, last_width, 3), dtype=np.uint8)
                panel = build_panel(
                    last_width, None, writer.count, detector,
                    config.collector.max_rms, config.collector.panel_height,
                )

            cv2.imshow(config.collector.window_name, np.vstack((last_display, panel)))

            key = get_key()
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                try:
                    capture_sample(robot, latest, writer, config)
                except Exception as e:
                    print("[SAVE FAILED]", repr(e))

            try:
                if cv2.getWindowProperty(config.collector.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        print("\nInterrupted, shutting down")

    finally:
        # Each device is released independently: one failing to close must not
        # leave the other one open.
        if camera_started:
            try:
                camera.stop()
            except Exception as e:
                print("[CLEANUP WARNING] Failed to stop camera:", repr(e))
        if robot_connected:
            try:
                robot.disconnect()
            except Exception as e:
                print("[CLEANUP WARNING] Failed to disconnect robot:", repr(e))
        cv2.destroyAllWindows()
        print(f"Finished. Samples collected: {writer.count}")
        # Only advertise a data file if one was actually written: an empty
        # session has no samples.json.
        if writer.count:
            print("Data file:", writer.json_path)


if __name__ == "__main__":
    main()
