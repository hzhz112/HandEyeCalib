from abc import ABC, abstractmethod
import numpy as np

def validate_transform_matrix(matrix):
    """Validate and copy the homogeneous transformation matrix provided by the robot.
    Invalid data will not be saved.
    """
    T = np.asarray(matrix, dtype=np.float64)

    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(
            "Robot pose must be a valid 4x4 homogeneous transformation matrix "
            "containing finite values."
        )

    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(
            "The last row of the homogeneous transformation matrix must be "
            "[0, 0, 0, 1]."
        )

    R = T[:3, :3]

    if (
        not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(R), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "The rotation matrix in the robot pose is not a valid SO(3) rotation."
        )

    return T.copy()


class RobotInterface(ABC):
    name = "UnknownRobot"

    @abstractmethod
    def connect(self) -> None:
        """Initialize and establish the robot connection.
        Raise an exception if the connection fails.
        """

    @abstractmethod
    def get_base_T_flange(self) -> np.ndarray:
        """Get the homogeneous transformation matrix from flange frame to robot base frame.

        Returns:
            np.ndarray:
                A 4x4 homogeneous transformation matrix.
                Translation unit: meters.
        """
        
    @abstractmethod
    def disconnect(self) -> None:
        """Close the robot connection.
        Calling this method multiple times should be safe.
        """