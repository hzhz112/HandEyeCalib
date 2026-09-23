"""Robot adapter selection.

The factory takes an explicit config object instead of reading environment
variables, so `config/handeye.yaml` is the single source of truth for how the
robot is reached.
"""

from __future__ import annotations

from calibration.config.loader import RobotConfig
from calibration.robot.robot_interface import RobotInterface


def create_robot(config: RobotConfig) -> RobotInterface:
    robot_type = config.type.lower().strip()

    if robot_type == "realman":
        from calibration.robot.realman_robot import RealManRobot

        return RealManRobot(ip=config.ip, port=config.port)

    # Adding a robot means writing the adapter, adding a branch here, and
    # adding the type to handeye.yaml. Nothing under apps/ needs to change.
    # if robot_type == "ur":
    #     from calibration.robot.ur_robot import URRobot
    #     return URRobot(ip=config.ip)

    raise ValueError(
        f"Unregistered robot type: {config.type!r}. "
        "Please register it in robot_factory.py."
    )
