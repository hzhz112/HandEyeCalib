"""The unit of data this tool collects.

One sample is a board pose in the color camera frame paired with the flange
pose in the robot base frame at approximately the same instant. The robot and
the camera are read one after the other, not synchronously, so the arm must be
stationary when a sample is taken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandEyeSample:
    """A single handeye sample.

    The attribute names are the internal ones and are deliberately spelled
    differently from the on-disk keys; see `to_dict`. `index` and `image_path`
    are assigned by `DatasetWriter.save`, so they are placeholders at
    construction time.
    """

    index: int
    # Host wall-clock time of the frame, from `time.time()`.
    camera_time: float
    # Midpoint of the robot pose read, from `time.time()`.
    robot_time: float
    # Camera device clock in ms. A different time base from the two above.
    device_timestamp_ms: float
    # flange -> robot base, 4x4, translation in metres.
    T_base_flange: np.ndarray
    # board -> color camera, 4x4, translation in metres.
    T_camera_board: np.ndarray
    image_path: str
    reprojection_error: float
    aruco_markers: int
    charuco_corners: int

    def to_dict(self) -> dict:
        """Serialize to the on-disk schema.

        The key names here are a frozen contract -- `solve_handeye.py` reads
        `index`, `T_base_flange`, `T_color_board`, `charuco_corners` and
        `reprojection_rms_px`. Renaming a key to match the attribute names
        above requires updating that script in the same change.

        Scalars are cast explicitly because numpy scalars are not JSON
        serializable, and the resulting TypeError would be swallowed by the
        save path's broad `except`, silently discarding every sample.
        """
        return {
            "index": int(self.index),
            "camera_host_time": float(self.camera_time),
            "robot_host_time": float(self.robot_time),
            "device_timestamp_ms": float(self.device_timestamp_ms),
            "T_base_flange": np.asarray(self.T_base_flange, dtype=np.float64).tolist(),
            "T_color_board": np.asarray(self.T_camera_board, dtype=np.float64).tolist(),
            "aruco_markers": int(self.aruco_markers),
            "charuco_corners": int(self.charuco_corners),
            "reprojection_rms_px": float(self.reprojection_error),
            "image": self.image_path,
        }
