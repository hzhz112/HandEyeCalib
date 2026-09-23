"""ChArUco board detection and board-pose estimation.

Everything ChArUco lives here: board construction, corner detection, solvePnP
and the reprojection error. The only thing this module needs from the outside
is a camera adapter, and it uses exactly two of its methods -- `get_intrinsics`
and `prepare_image_points` -- so it imports no vendor SDK and can be tested
against a fake camera with no hardware attached.

Distortion handling deliberately stays inside the camera adapter. The detector
must never de-distort points itself, and must never take an intrinsics matrix
from configuration: `K` is only ever read from the same adapter that
`prepare_image_points` used, because a mismatch between the two produces a
board pose that is self-consistent, wrong, and has a deceptively low RMS.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera.camera_interface import CameraInterface


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Geometry of a ChArUco board. All lengths are in metres."""

    squares_x: int = 14
    squares_y: int = 9
    square_length: float = 0.020
    marker_length: float = 0.015
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False


@dataclass(frozen=True, eq=False)  # eq=False: ndarray fields would break __eq__/__hash__
class BoardDetection:
    """Outcome of one detection attempt.

    `corner_count` and `marker_count` are counts, not point arrays. Keeping
    them separate from `points` is deliberate: `corner_count` is written to
    `samples.json` as `charuco_corners` and compared numerically downstream,
    so an array landing in that field would abort the solver with a message
    that looks like "not enough corners".
    """

    # True only when a full board pose was solved.
    success: bool
    # Board -> color camera, 4x4. None unless `success`.
    T_camera_board: np.ndarray | None
    # Reprojection RMS in pixels. None unless `success`.
    rms: float | None
    # Raw detected corners, shape (N, 2) or (0, 2), for drawing only.
    #
    # These are the undecorated detectBoard output. Under a distortion model such
    # as inverse_brown_conrady they are NOT the points solvePnP consumed, so
    # drawing the prepared points instead would misplace every overlay marker.
    # Populated whenever detectBoard returned a well-shaped result, including on
    # the rejection paths below, so the operator can still see what was found.
    points: np.ndarray
    corner_count: int
    marker_count: int

    @classmethod
    def empty(
        cls,
        corner_count: int = 0,
        marker_count: int = 0,
        points: np.ndarray | None = None,
    ) -> "BoardDetection":
        if points is None:
            points = np.zeros((0, 2), dtype=np.float64)
        return cls(
            success=False,
            T_camera_board=None,
            rms=None,
            points=points,
            corner_count=int(corner_count),
            marker_count=int(marker_count),
        )


class CharucoDetector:
    """Detects a ChArUco board and estimates its pose in the color camera frame.

    Named after `cv2.aruco.CharucoDetector` on purpose. The cv2 type is always
    spelled fully qualified inside this module so the two never get confused.
    """

    def __init__(self, spec: CharucoBoardSpec, min_corners: int = 30) -> None:
        self.spec = spec
        self.min_corners = int(min_corners)

        if not hasattr(cv2.aruco, spec.dictionary):
            raise ValueError(
                f"Unknown ArUco dictionary: {spec.dictionary!r}. "
                "Check the `dictionary` key in the board config."
            )
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))

        # Order matters: setLegacyPattern must be applied to the board before
        # the detector is constructed from it.
        board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_length,
            spec.marker_length,
            dictionary,
        )
        board.setLegacyPattern(bool(spec.legacy_pattern))

        self._board = board
        self._detector = cv2.aruco.CharucoDetector(board)
        self._board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)

        # Derived from the board rather than hardcoded, so changing the board
        # geometry cannot leave the on-screen totals reporting obsolete numbers.
        self.corner_total = int(len(self._board_points))
        self.marker_total = int(len(board.getIds()))

    @property
    def board_points(self) -> np.ndarray:
        """Object points of every corner, in board coordinates (metres)."""
        return self._board_points

    def detect(self, image: np.ndarray, camera: CameraInterface) -> BoardDetection:
        """Detect the board in a BGR frame and solve its pose.

        Returns a `BoardDetection` in every case; a failed detection is not an
        exception. `camera` supplies both the intrinsics and the distortion
        handling, and must be the same adapter the caller uses elsewhere.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _marker_corners, marker_ids = self._detector.detectBoard(gray)

        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        corner_count = 0 if ids is None else int(len(np.asarray(ids).reshape(-1)))

        if corners is None or ids is None:
            return BoardDetection.empty(corner_count, marker_count)

        points = np.asarray(corners, dtype=np.float64)
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)

        if points.size % 2:
            return BoardDetection.empty(corner_count, marker_count)

        points = points.reshape(-1, 2)
        if len(points) != len(ids):
            return BoardDetection.empty(corner_count, marker_count)

        # From here on the raw corners are kept even when detection is rejected
        # below: the operator overlay still needs whatever was found.
        def rejected() -> BoardDetection:
            return BoardDetection.empty(len(ids), marker_count, points)

        if (
            len(ids) < self.min_corners
            or np.any(ids < 0)
            or np.any(ids >= len(self._board_points))
        ):
            return rejected()
        if len(np.unique(ids)) != len(ids):
            return rejected()

        object_points = np.ascontiguousarray(self._board_points[ids], dtype=np.float64)

        intrinsics = camera.get_intrinsics()
        image_points, dist_coeffs = camera.prepare_image_points(points)
        image_points = np.ascontiguousarray(image_points, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics.K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or float(tvec[2, 0]) <= 0:
            return rejected()

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, intrinsics.K, dist_coeffs
        )
        error = projected.reshape(-1, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3], T[:3, 3] = R, tvec.reshape(3)

        return BoardDetection(
            success=True,
            T_camera_board=T,
            rms=rms,
            points=points,
            corner_count=int(len(ids)),
            marker_count=marker_count,
        )
