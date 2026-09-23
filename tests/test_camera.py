"""Camera adapter tests.

These need the RealSense attached. When it is not, they skip instead of
failing, and they never block: `start()` is the only call that can stall on
hardware, and it is wrapped.
"""

from __future__ import annotations

import time

import numpy as np

from camera.camera_factory import create_camera
from camera.camera_interface import DISTORTION_MODELS, validate_intrinsics
from config.loader import CameraConfig, load_config
from tests.support import main, skip


def _started_camera():
    """Open the configured camera, or skip if there is no device."""
    try:
        config = load_config().camera
    except Exception:
        config = CameraConfig()

    camera = create_camera(config)
    try:
        camera.start()
    except Exception as error:
        skip(f"no camera available: {error!r}")
    return camera


def test_reports_valid_intrinsics_and_bgr_frames():
    camera = _started_camera()
    try:
        intrinsics = camera.get_intrinsics()
        assert intrinsics.K.shape == (3, 3)
        assert intrinsics.fps > 0
        assert intrinsics.distortion_model in DISTORTION_MODELS
        # Adapters return validate_intrinsics(...), so this must not raise.
        assert validate_intrinsics(intrinsics) is not None

        frame = camera.get_frame(timeout_ms=2000)
        assert frame is not None, "expected a frame within 2 s"

        # The BGR uint8 contract the detector and cv2.imwrite both rely on.
        assert frame.image.dtype == np.uint8
        assert frame.image.shape == (intrinsics.height, intrinsics.width, 3)

        # The staleness gate depends on monotonic_time being a monotonic clock
        # reading taken recently. A wall-clock value would look plausible here
        # and still break the gate, so assert the age explicitly.
        assert 0.0 <= time.monotonic() - frame.monotonic_time < 1.0
        assert frame.host_time > 0
        assert frame.device_timestamp_ms > 0
    finally:
        camera.stop()


def test_prepare_image_points_contract():
    camera = _started_camera()
    try:
        intrinsics = camera.get_intrinsics()
        points = np.array([[100.0, 100.0], [320.0, 240.0], [500.0, 400.0]])

        image_points, dist_coeffs = camera.prepare_image_points(points)

        assert image_points.shape == (3, 2)
        assert np.isfinite(image_points).all()
        assert np.asarray(dist_coeffs).reshape(-1).shape == (5,)

        if intrinsics.distortion_model == "inverse_brown_conrady":
            # Under this model the adapter de-distorts point by point and hands
            # solvePnP zero distortion, so zeros are the observable signature
            # that the branch was taken. Note that a unit reporting zero lens
            # coefficients (as this D435 does) makes the de-distortion an
            # identity map, so the points themselves barely move -- asserting
            # that they differ would be asserting something about the lens,
            # not about the adapter.
            assert np.allclose(np.asarray(dist_coeffs).reshape(-1), 0.0)
            assert np.allclose(image_points, points, atol=1e-3)
    finally:
        camera.stop()


def test_stop_is_idempotent_and_use_after_stop_is_an_error():
    camera = _started_camera()
    camera.stop()
    camera.stop()  # must be safe

    try:
        camera.get_frame(timeout_ms=10)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError from get_frame after stop()")


def test_short_timeout_is_not_an_exception():
    """A missing frame is reported as None, never as a raised error."""
    camera = _started_camera()
    try:
        camera.get_frame(timeout_ms=1)
    finally:
        camera.stop()


if __name__ == "__main__":
    main(dict(globals()))
