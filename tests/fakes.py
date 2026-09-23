"""Stand-ins for the device layer, so the domain tests need no hardware.

Importing this module must not pull in a vendor SDK: `camera_interface` only
needs numpy, which is the whole point of the adapter split.
"""

from __future__ import annotations

import cv2
import numpy as np

from board.charuco_detector import CharucoBoardSpec
from camera.camera_interface import CameraIntrinsics

# Real D435 color intrinsics, so the synthetic tests use realistic numbers.
D435_K = np.array(
    [
        [605.579, 0.0, 322.194],
        [0.0, 605.054, 253.386],
        [0.0, 0.0, 1.0],
    ]
)
D435_WIDTH, D435_HEIGHT, D435_FPS = 640, 480, 15


class FakeCamera:
    """A `CameraInterface` with no hardware behind it.

    `shift` offsets whatever `prepare_image_points` returns, independently of
    the raw corners the detector found. That is how a test proves the detector
    reports the raw corners rather than the de-distorted ones.
    """

    name = "FakeCamera"

    def __init__(self, distortion_model: str = "none", shift=(0.0, 0.0)) -> None:
        self._intrinsics = CameraIntrinsics(
            K=D435_K.copy(),
            dist_coeffs=np.zeros(5),
            width=D435_WIDTH,
            height=D435_HEIGHT,
            fps=D435_FPS,
            distortion_model=distortion_model,
        )
        self._shift = np.asarray(shift, dtype=np.float64)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_frame(self, timeout_ms: int = 1000):
        return None

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def prepare_image_points(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return points + self._shift, np.zeros(5)


def render_board(spec: CharucoBoardSpec, size=(1400, 900), margin: int = 20) -> np.ndarray:
    """Render a board to a BGR image, roughly as a camera would see it.

    `generateImage` returns a single-channel image, while the detector always
    converts with COLOR_BGR2GRAY. The GRAY2BGR conversion here is required:
    without it `detect()` fails with "Invalid number of channels".
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length,
        spec.marker_length,
        dictionary,
    )
    board.setLegacyPattern(bool(spec.legacy_pattern))
    return cv2.cvtColor(board.generateImage(size, marginSize=margin), cv2.COLOR_GRAY2BGR)
