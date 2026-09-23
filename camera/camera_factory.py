"""Camera adapter selection.

The factory takes an explicit config object instead of reading environment
variables, so `config/handeye.yaml` is the single source of truth for how the
camera is opened.
"""

from __future__ import annotations

from calibration.camera.camera_interface import CameraInterface
from calibration.config.loader import CameraConfig

# Spellings accepted for each registered adapter, beyond its canonical name.
_ALIASES = {"d435": "realsense_d435"}


def create_camera(config: CameraConfig) -> CameraInterface:
    camera_type = config.type.lower().strip()
    camera_type = _ALIASES.get(camera_type, camera_type)

    if camera_type == "realsense_d435":
        from calibration.camera.realsense_camera import RealSenseD435

        return RealSenseD435(
            width=config.width,
            height=config.height,
            fps=config.fps,
        )

    # Adding a camera means writing the adapter, adding a branch here, and
    # adding the type to handeye.yaml. Nothing under apps/ needs to change.
    # if camera_type == "usb":
    #     from calibration.camera.usb_camera import USBCamera
    #     return USBCamera(index=config.index)

    raise ValueError(
        f"Unregistered camera type: {config.type!r}. "
        "Please register it in camera_factory.py."
    )
