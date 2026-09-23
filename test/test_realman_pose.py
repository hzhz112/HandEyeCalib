from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_thread_mode_e,
)
import numpy as np

def get_realman_pose(arm):
    """
    Read the current end-effector pose of the RealMan arm.

    arm: a connected RoboticArm instance

    Returns the raw pose as reported by the SDK.
    """

    code, state = arm.rm_get_current_arm_state()

    if code != 0:
        raise RuntimeError(
            f"Failed to read the RealMan end-effector pose, error code: {code}"
        )

    pose = state["pose"]

    return pose

def realman_pose_to_matrix(pose):
    pose = np.asarray(
        pose,
        dtype=np.float64,
    ).reshape(-1)

    if pose.size != 6:
        raise ValueError(
            f"RealMan pose must have 6 elements, got: {pose}"
        )

    if not np.isfinite(pose).all():
        raise ValueError("RealMan pose contains NaN or Inf")

    x, y, z, rx, ry, rz = pose

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ])

    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])

    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)

    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


if __name__ == "__main__":

    # Set this to your RealMan arm's actual IP.
    ROBOT_IP = "192.168.1.18"
    ROBOT_PORT = 8080

    arm = RoboticArm(
        rm_thread_mode_e.RM_TRIPLE_MODE_E
    )

    try:
        print("Connecting to RealMan...")

        handle = arm.rm_create_robot_arm(
            ROBOT_IP,
            ROBOT_PORT
        )

        if handle is None or handle.id <= 0:
            raise RuntimeError(
                "RealMan connection failed; check the IP and the network"
            )

        print(f"RealMan connected, handle.id = {handle.id}")

        # Read the current end-effector pose.
        pose = get_realman_pose(arm)

        print("\n========== Current end-effector pose ==========")
        print(pose)
        print("================================================")

    finally:
        arm.rm_delete_robot_arm()
        print("RealMan connection closed")
