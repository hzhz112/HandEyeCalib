import time
import numpy as np

from camera.camera_interface import (
    CameraFrame,
    CameraInterface,
    CameraIntrinsics,
    validate_intrinsics,
)

class RealSenseD435(CameraInterface):
    name = "RealSenseD435"

    def __init__(self, width=640, height=480, fps=15):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._rs = None
        self._pipeline = None
        self._profile = None
        self._intrinsics = None
        self._raw_intrinsics = None
        self._started = False

    def start(self):
        if self._started:
            return

        rs = self._require_rs()

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        self._pipeline = pipeline
        self._profile = pipeline.start(config)
        self._started = True

        try:
            self._intrinsics = self._read_intrinsics()
        except Exception:
            self.stop()
            raise

    def stop(self):
        pipeline = self._pipeline

        self._pipeline = None
        self._profile = None
        self._intrinsics = None
        self._raw_intrinsics = None
        self._started = False

        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def get_frame(self, timeout_ms=1000):
        self._require_started()

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None

        color = frames.get_color_frame()

        if not color:
            return None

        return CameraFrame(
            image=np.asanyarray(color.get_data()).copy(),
            host_time=time.time(),
            monotonic_time=time.monotonic(),
            device_timestamp_ms=float(color.get_timestamp()),
        )

    def get_intrinsics(self):
        self._require_started()
        return self._intrinsics

    def prepare_image_points(self, points):
        self._require_started()

        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

        if self._intrinsics.distortion_model != "inverse_brown_conrady":
            return points, self._intrinsics.dist_coeffs

        return self._undistort_inverse_brown(points)

    def _undistort_inverse_brown(self, points):

        rs = self._rs
        K = self._intrinsics.K

        result = []

        for u, v in points:
            x, y, z = rs.rs2_deproject_pixel_to_point(
                self._raw_intrinsics,
                [float(u), float(v)],
                1.0,
            )

            result.append(
                [
                    K[0, 0] * x / z + K[0, 2],
                    K[1, 1] * y / z + K[1, 2],
                ]
            )

        return np.asarray(result), np.zeros(5)

    def _read_intrinsics(self):

        rs = self._rs

        raw = (
            self._profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self._raw_intrinsics = raw

        K = np.array(
            [
                [raw.fx, 0, raw.ppx],
                [0, raw.fy, raw.ppy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        return validate_intrinsics(
            CameraIntrinsics(
                K=K,
                dist_coeffs=np.asarray(raw.coeffs),
                width=raw.width,
                height=raw.height,
                fps=self.fps,
                distortion_model=self._convert_model(raw.model),
            )
        )

    def _convert_model(self, model):

        rs = self._rs

        mapping = {
            rs.distortion.none: "none",
            rs.distortion.brown_conrady: "brown_conrady",
            rs.distortion.inverse_brown_conrady:
                "inverse_brown_conrady",
        }

        if model not in mapping:
            raise RuntimeError(
                f"Unsupported distortion model: {model}"
            )

        return mapping[model]

    def _require_rs(self):

        if self._rs is None:
            import pyrealsense2 as rs
            self._rs = rs

        return self._rs

    def _require_started(self):

        if not self._started:
            raise RuntimeError(
                "RealSenseD435 is not started."
            )