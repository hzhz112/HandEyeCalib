from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

DISTORTION_NONE = "none"
DISTORTION_BROWN_CONRADY = "brown_conrady"
DISTORTION_INVERSE_BROWN_CONRADY = "inverse_brown_conrady"

DISTORTION_MODELS = (
    DISTORTION_NONE,
    DISTORTION_BROWN_CONRADY,
    DISTORTION_INVERSE_BROWN_CONRADY,
)


@dataclass(eq=False)  # eq=False: ndarray fields would break __eq__
class CameraFrame:
    """
    A color image frame with timestamp information.
    """

    image: np.ndarray
    """
    BGR uint8 image with shape (H, W, 3).
    """

    host_time: float
    """
    Host timestamp from time.time().
    """

    monotonic_time: float
    """
    Monotonic timestamp for frame age checking.
    """

    device_timestamp_ms: float
    """
    Camera internal timestamp in milliseconds.
    """


@dataclass(eq=False)  # eq=False: ndarray fields would break __eq__
class CameraIntrinsics:
    """
    Camera intrinsic parameters.
    """

    K: np.ndarray
    """
    3x3 camera intrinsic matrix.
    """

    dist_coeffs: np.ndarray
    """
    Distortion coefficients.
    """

    width: int
    height: int
    fps: int

    distortion_model: str
    """
    One of DISTORTION_MODELS.
    """


def validate_intrinsics(intrinsics: CameraIntrinsics) -> CameraIntrinsics:
    """Validate the intrinsic parameters reported by a camera adapter.

    Rejects anything that would corrupt solvePnP's result while still looking
    plausible, and returns a copy so callers cannot mutate the adapter's state.
    Adapters should return `validate_intrinsics(...)` from `get_intrinsics()`.

    Raises:
        ValueError: if a parameter is malformed.
    """
    K = np.asarray(intrinsics.K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Camera intrinsic matrix K must be a finite 3x3 matrix.")

    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(
            f"Camera focal lengths must be positive, got fx={K[0, 0]}, fy={K[1, 1]}."
        )

    if intrinsics.width <= 0 or intrinsics.height <= 0 or intrinsics.fps <= 0:
        raise ValueError(
            f"Camera stream size and frame rate must be positive, got "
            f"{intrinsics.width}x{intrinsics.height}@{intrinsics.fps}."
        )

    if not (0 <= K[0, 2] < intrinsics.width and 0 <= K[1, 2] < intrinsics.height):
        raise ValueError(
            f"Principal point ({K[0, 2]}, {K[1, 2]}) lies outside the "
            f"{intrinsics.width}x{intrinsics.height} image."
        )

    dist_coeffs = np.asarray(intrinsics.dist_coeffs, dtype=np.float64).reshape(-1)
    if not np.isfinite(dist_coeffs).all():
        raise ValueError("Distortion coefficients must be finite.")

    if not str(intrinsics.distortion_model).strip():
        raise ValueError("Distortion model name must be a non-empty string.")

    return CameraIntrinsics(
        K=K.copy(),
        dist_coeffs=dist_coeffs.copy(),
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fps=int(intrinsics.fps),
        distortion_model=intrinsics.distortion_model,
    )


class CameraInterface(ABC):

    name = "UnknownCamera"

    @abstractmethod
    def start(self) -> None:
        """
        Start camera streaming.
        """


    @abstractmethod
    def get_frame(
        self,
        timeout_ms: int = 1000
    ) -> CameraFrame | None:
        """
        Get the latest color frame.

        Returns:
            CameraFrame or None if timeout.
        """

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """
        Return camera intrinsic parameters.
        """

    @abstractmethod
    def prepare_image_points(
        self,
        points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert detected pixel points into OpenCV solvePnP format.

        The returned pair is the `(image_points, dist_coeffs)` argument pair of
        `cv2.solvePnP(obj, image_points, K, dist_coeffs)` and
        `cv2.projectPoints(...)`, where `K` is `get_intrinsics().K`.

        The returned image_points are NOT the input points in general: under
        inverse_brown_conrady they are de-distorted. Callers drawing on the
        original image must keep using the points they passed in.

        Returns:
            image_points:
                Points used by solvePnP.

            dist_coeffs:
                Distortion coefficients.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop camera streaming.
        """
