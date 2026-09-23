import numpy as np
from scipy.spatial.transform import Rotation
from calibration.robot.robot_interface import RobotInterface, validate_transform_matrix

class RealManRobot(RobotInterface):
    name = "RealMan"

    def __init__(self, ip: str, port: int = 8080):
        self.ip = ip
        self.port = int(port)
        self._arm = None

    def connect(self):
        if self._arm is not None:
            return

        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        try:
            handle = arm.rm_create_robot_arm(self.ip, self.port)
            if handle is None or getattr(handle, "id", 0) <= 0:
                raise RuntimeError(
                    f"Failed to connect RealMan: {self.ip}:{self.port}"
                )
        except Exception:
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                pass
            raise

        self._arm = arm

    def get_base_T_flange(self) -> np.ndarray:
        if self._arm is None:
            raise RuntimeError("RealMan is not connected. Call connect() first.")

        pose = np.asarray(self._read_pose(), dtype=np.float64).reshape(-1)

        if pose.size != 6 or not np.isfinite(pose).all():
            raise ValueError(f"Invalid RealMan pose: {pose}")

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_euler(
            "xyz", pose[3:], degrees=False
        ).as_matrix()
        T[:3, 3] = pose[:3]

        return validate_transform_matrix(T)

    def disconnect(self):
        if self._arm is None:
            return
        try:
            self._arm.rm_delete_robot_arm()
        finally:
            self._arm=None

    def _read_pose(self):
        """Read the current end-effector pose as [x, y, z, rx, ry, rz].

        Previously imported from calibration.test.test_realman_pose, inlined
        here because production code must not depend on a test module.
        """
        code, state = self._arm.rm_get_current_arm_state()

        if code != 0:
            raise RuntimeError(f"Failed to read RealMan end-effector pose. Error code: {code}")

        return state["pose"]
