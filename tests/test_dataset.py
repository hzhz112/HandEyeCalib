"""`samples.json` contract tests.

The schema is read by `solve_handeye.py`, so it is frozen. The
failures these guard against are quiet ones: a numpy scalar that cannot be
serialized, or an array landing in a field the solver compares numerically,
both of which the save path's broad `except` turns into "sample silently
discarded" rather than a visible error.

No hardware: a fake camera supplies the metadata and a temporary directory
stands in for the session folder.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np

from config.loader import BoardConfig
from dataset.sample import HandEyeSample
from dataset.writer import DatasetWriter
from tests.fakes import FakeCamera
from tests.support import main

# The keys solve_handeye.py reads. Renaming any of these breaks the solver.
SOLVER_KEYS = {
    "index",
    "T_base_flange",
    "T_color_board",
    "charuco_corners",
    "reprojection_rms_px",
}

SAMPLE_KEYS = {
    "index",
    "camera_host_time",
    "robot_host_time",
    "device_timestamp_ms",
    "T_base_flange",
    "T_color_board",
    "aruco_markers",
    "charuco_corners",
    "reprojection_rms_px",
    "image",
}

TOP_LEVEL_KEYS = {"camera", "robot", "board", "coordinate_convention", "samples"}


class _FakeRobot:
    name = "FakeRobot"


def _sample() -> HandEyeSample:
    return HandEyeSample(
        index=0,
        camera_time=1.0,
        robot_time=2.0,
        device_timestamp_ms=3.0,
        T_base_flange=np.eye(4),
        T_camera_board=np.eye(4),
        image_path="",
        reprojection_error=0.25,
        aruco_markers=63,
        charuco_corners=104,
    )


def _image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_sample_serializes_to_the_frozen_schema():
    payload = _sample().to_dict()
    assert set(payload) == SAMPLE_KEYS
    assert SOLVER_KEYS <= set(payload)


def test_counts_stay_ints():
    """solve_handeye.py compares `charuco_corners` numerically. If an array
    ever reaches that field the comparison raises, and the solver reports
    insufficient data while printing a message about corner counts."""
    payload = _sample().to_dict()
    assert isinstance(payload["charuco_corners"], int)
    assert isinstance(payload["aruco_markers"], int)
    assert isinstance(payload["index"], int)


def test_payload_is_json_serializable():
    """numpy scalars are not JSON serializable. The resulting TypeError would
    be caught by the save path's broad except and lose the sample."""
    json.dumps(_sample().to_dict())


def test_writer_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        writer = DatasetWriter(tmp, BoardConfig(), FakeCamera(), _FakeRobot())
        assert writer.count == 0

        writer.save(_sample(), _image())
        saved = writer.save(_sample(), _image())

        assert writer.count == 2
        assert saved.index == 2
        assert saved.image_path == "images/sample_002.png"
        assert (writer.image_dir / "sample_001.png").is_file()
        assert (writer.image_dir / "sample_002.png").is_file()

        data = json.loads(writer.json_path.read_text(encoding="utf-8"))
        assert set(data) == TOP_LEVEL_KEYS
        assert len(data["samples"]) == 2
        assert set(data["samples"][0]) == SAMPLE_KEYS
        assert SOLVER_KEYS <= set(data["samples"][1])

        # Forward slashes: downstream tools join this onto the session dir.
        assert data["samples"][0]["image"] == "images/sample_001.png"
        assert "\\" not in data["samples"][0]["image"]

        assert data["camera"]["type"] == "FakeCamera"
        assert data["camera"]["image_color_order"] == "BGR"
        assert data["camera"]["width"] == 640
        assert data["robot"]["type"] == "FakeRobot"
        assert data["robot"]["pose_frame"] == "flange"
        assert data["board"]["dictionary"] == "DICT_5X5_100"


def test_failed_save_does_not_consume_an_index():
    """The index comes from the committed count, so a failure leaves no gap
    and no name collision on the next attempt."""
    with tempfile.TemporaryDirectory() as tmp:
        writer = DatasetWriter(tmp, BoardConfig(), FakeCamera(), _FakeRobot())

        try:
            writer.save(_sample(), None)  # cv2.imwrite cannot write None
        except Exception:
            pass
        else:
            raise AssertionError("expected the failed write to raise")

        assert writer.count == 0
        assert not writer.json_path.exists(), "nothing should have been committed"

        saved = writer.save(_sample(), _image())
        assert saved.index == 1
        assert saved.image_path == "images/sample_001.png"


if __name__ == "__main__":
    main(dict(globals()))
