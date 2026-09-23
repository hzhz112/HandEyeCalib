"""Session directory layout and persistence for collected samples.

This module is the only place in the package allowed to call `cv2.imwrite` or
`json.dump`. Keeping the on-disk schema in one file is what lets the
application layer stay ignorant of the JSON format, and what makes the schema
testable without running the collector.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dataset.sample import HandEyeSample

if TYPE_CHECKING:
    # Annotations only: the writer reads `.name` and `get_intrinsics()` off
    # whatever it is handed, so it needs no import at runtime.
    from camera.camera_interface import CameraInterface
    from config.loader import BoardConfig
    from robot.robot_interface import RobotInterface


class DatasetWriter:
    """Writes one collection session: the images plus a `samples.json` manifest.

    The session directory is created eagerly in the constructor, before any
    device is opened. That matches the previous behavior, where an aborted run
    still left an empty session directory behind for inspection.

    Note that two runs started within the same second share a session name and
    the second one overwrites the first one's manifest. That is pre-existing
    behavior; it is documented rather than changed here.
    """

    def __init__(
        self,
        output_root: str | Path,
        board_config: "BoardConfig",
        camera: "CameraInterface",
        robot: "RobotInterface",
        session_name: str | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.board_config = board_config
        self.camera = camera
        self.robot = robot

        name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = self.output_root / name
        self.image_dir = self.session_dir / "images"
        self.json_path = self.session_dir / "samples.json"

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._samples: list[HandEyeSample] = []

    @property
    def count(self) -> int:
        """Number of samples successfully committed to disk."""
        return len(self._samples)

    def save(self, sample: HandEyeSample, image: np.ndarray) -> HandEyeSample:
        """Persist one sample, returning it with `index` and `image_path` filled in.

        The index is derived from the committed count rather than from a
        pre-incremented counter, so a failure part way through does not consume
        an index and leave a permanent gap in the image numbering.
        """
        index = len(self._samples) + 1
        image_name = f"sample_{index:03d}.png"

        # Forward slashes: this string is stored in JSON and joined onto the
        # session directory by downstream tools on any platform.
        image_path = f"images/{image_name}"

        if not cv2.imwrite(str(self.image_dir / image_name), image):
            raise RuntimeError(f"Failed to write image: {self.image_dir / image_name}")

        sample.index = index
        sample.image_path = image_path
        self._samples.append(sample)
        self._flush()
        return sample

    def _payload(self) -> dict:
        """Build the full `samples.json` document."""
        intrinsics = self.camera.get_intrinsics()

        return {
            "camera": {
                "type": self.camera.name,
                "stream": "color",
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fps": int(intrinsics.fps),
                "K": np.asarray(intrinsics.K, dtype=np.float64).tolist(),
                # The raw coefficients reported by the SDK, NOT the zeros that
                # solvePnP is given for inverse_brown_conrady. Confusing, but
                # it is the existing schema; do not "fix" it here.
                "distortion_coefficients": np.asarray(
                    intrinsics.dist_coeffs, dtype=np.float64
                ).tolist(),
                "distortion_model": intrinsics.distortion_model,
                "image_color_order": "BGR",
            },
            "robot": {
                "type": self.robot.name,
                "pose_frame": "flange",
                "base_frame": "robot_base",
            },
            "board": {
                "squares_x": int(self.board_config.squares_x),
                "squares_y": int(self.board_config.squares_y),
                "square_length_m": float(self.board_config.square_length),
                "marker_length_m": float(self.board_config.marker_length),
                "dictionary": self.board_config.dictionary,
                "legacy_pattern": bool(self.board_config.legacy_pattern),
            },
            "coordinate_convention": {
                "T_base_flange": "flange_to_robot_base",
                "T_color_board": "board_to_color_camera",
                "position_unit": "m",
                "rotation": "3x3 rotation matrix (right-handed)",
            },
            "samples": [sample.to_dict() for sample in self._samples],
        }

    def _flush(self) -> None:
        """Rewrite the manifest atomically.

        Writing to a temporary file and replacing means an interrupted write
        cannot leave a truncated `samples.json` behind.
        """
        temp_path = self.json_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self._payload(), handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.json_path)
