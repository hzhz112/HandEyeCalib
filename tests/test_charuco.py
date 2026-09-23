"""ChArUco detection tests.

No hardware: every case runs against a rendered board and a fake camera. The
pre-refactor implementation in `calibration/collect_handeye_data.py` is used as
an oracle, because the move into `board/charuco_detector.py` must not change
the math by even one ulp.
"""

from __future__ import annotations

import numpy as np

from calibration.board.charuco_detector import CharucoBoardSpec, CharucoDetector
from calibration.tests.fakes import D435_HEIGHT, D435_K, D435_WIDTH, FakeCamera, render_board
from calibration.tests.support import main, skip

SPEC = CharucoBoardSpec()
EXPECTED_CORNERS = 104
EXPECTED_MARKERS = 63


class _LegacyIntrinsics:
    """Mimics the pyrealsense2 intrinsics object the legacy function expects."""

    def __init__(self, model, coeffs) -> None:
        self.model = model
        self.coeffs = coeffs


def _legacy_oracle():
    """Import the monolith, or skip if its dependencies are unavailable."""
    try:
        import pyrealsense2 as rs
        import calibration.collect_handeye_data as legacy
    except ImportError as error:
        skip(f"legacy oracle unavailable: {error!r}")
    return legacy, rs


def test_detects_full_board():
    detection = CharucoDetector(SPEC).detect(render_board(SPEC), FakeCamera())

    assert detection.success
    assert detection.corner_count == EXPECTED_CORNERS
    assert detection.marker_count == EXPECTED_MARKERS
    assert detection.points.shape == (EXPECTED_CORNERS, 2)
    assert detection.rms < 1.0

    T = detection.T_camera_board
    assert T.shape == (4, 4)
    assert np.isfinite(T).all()
    assert np.allclose(T[3], [0, 0, 0, 1])

    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "rotation is not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    # The board sits in front of the camera, not behind it.
    assert T[2, 3] > 0


def test_matches_the_legacy_implementation():
    """The strongest guarantee that this refactor did not alter the math."""
    legacy, rs = _legacy_oracle()
    image = render_board(SPEC)

    new = CharucoDetector(SPEC).detect(image, FakeCamera())
    old = legacy.detect_board(
        image, _LegacyIntrinsics(rs.distortion.none, (0.0,) * 5), D435_K
    )

    assert new.marker_count == old["markers"]
    assert new.corner_count == old["corners"]
    # Exact equality on purpose: this is a port, not a reimplementation.
    assert new.rms == old["rms"]
    assert np.array_equal(new.points, old["points"])
    assert np.array_equal(new.T_camera_board, old["T"])


def test_blank_image_is_rejected():
    blank = np.zeros((D435_HEIGHT, D435_WIDTH, 3), dtype=np.uint8)
    detection = CharucoDetector(SPEC).detect(blank, FakeCamera())

    assert not detection.success
    assert detection.T_camera_board is None
    assert detection.rms is None
    assert detection.corner_count == 0
    assert detection.marker_count == 0
    assert detection.points.shape == (0, 2)


def test_points_are_raw_corners_not_prepared_ones():
    """The overlay draws `points`. Under a distortion model those must be the
    raw detectBoard output, not the de-distorted coordinates solvePnP saw."""
    image = render_board(SPEC)

    plain = CharucoDetector(SPEC).detect(image, FakeCamera(shift=(0.0, 0.0)))
    shifted = CharucoDetector(SPEC).detect(
        image, FakeCamera(distortion_model="brown_conrady", shift=(5.0, 5.0))
    )

    assert plain.success and shifted.success
    # The shift changes what solvePnP was given, so the recovered pose differs...
    assert not np.allclose(plain.T_camera_board, shifted.T_camera_board)
    # ...but the corners reported for drawing are untouched.
    assert np.allclose(plain.points, shifted.points)


def test_points_survive_a_rejection():
    """A rejected detection must still report what it found, otherwise the
    operator loses the aiming overlay exactly when it is needed most."""
    detection = CharucoDetector(SPEC, min_corners=200).detect(
        render_board(SPEC), FakeCamera()
    )

    assert not detection.success
    assert detection.T_camera_board is None
    assert detection.corner_count == EXPECTED_CORNERS
    assert detection.points.shape == (EXPECTED_CORNERS, 2)


def test_board_totals_are_derived_from_the_board():
    detector = CharucoDetector(SPEC)
    assert detector.corner_total == EXPECTED_CORNERS
    assert detector.marker_total == EXPECTED_MARKERS
    assert detector.board_points.shape == (EXPECTED_CORNERS, 3)


def test_unknown_dictionary_is_rejected_at_construction():
    try:
        CharucoDetector(CharucoBoardSpec(dictionary="DICT_NOT_A_REAL_BOARD"))
    except ValueError as error:
        assert "DICT_NOT_A_REAL_BOARD" in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown dictionary")


if __name__ == "__main__":
    main(dict(globals()))
