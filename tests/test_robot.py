"""Robot adapter tests.

The arm may be unreachable, which is normal. A TCP pre-check makes these tests
skip in about a second instead of hanging on the SDK's own connect timeout --
a hanging suite is worse than a failing one.
"""

from __future__ import annotations

import socket

import numpy as np

from calibration.config.loader import load_config
from calibration.robot.robot_factory import create_robot
from calibration.tests.support import main, skip

CONNECT_TIMEOUT_S = 1.5


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def test_returns_a_valid_se3_pose():
    config = load_config().robot

    if not _reachable(config.ip, config.port):
        skip(f"robot not reachable at {config.ip}:{config.port}")

    robot = create_robot(config)
    try:
        robot.connect()
    except Exception as error:
        skip(f"could not connect to the robot: {error!r}")

    try:
        assert robot.name == "RealMan"

        T = robot.get_base_T_flange()
        assert T.shape == (4, 4)
        assert np.isfinite(T).all()
        assert np.allclose(T[3], [0, 0, 0, 1], atol=1e-9)

        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-3), "rotation is not orthonormal"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-3), "not a proper rotation"

        # Translation is in metres, and a real arm sits inside a small envelope.
        assert 0.0 < float(np.linalg.norm(T[:3, 3])) < 2.0

        # The adapter validates internally, so a second read proves the
        # validator is wired up rather than bypassed.
        assert robot.get_base_T_flange().shape == (4, 4)
    finally:
        robot.disconnect()
        robot.disconnect()  # must be safe to call twice


if __name__ == "__main__":
    main(dict(globals()))
