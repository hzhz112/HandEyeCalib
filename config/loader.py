"""Typed configuration for the handeye collector.

This is a leaf layer: it imports nothing from `camera/`, `robot/`, `board/`,
`dataset/` or `apps/`, so every other layer is free to depend on it. That is
what lets the factories take a config object instead of reading environment
variables.

Note the two thresholds shared with the solver: `min_corners` and `max_rms`
also exist in `calibration/solve_handeye.py`. Raising `max_rms` here means the
collector saves samples that the solver will silently discard, so change both
together.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# .../calibration -- relative output paths resolve against this, so the
# location of the data does not depend on the working directory.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "handeye.yaml"

_SECTIONS = ("camera", "robot", "board", "collector")


@dataclass(frozen=True)
class CameraConfig:
    type: str = "realsense_d435"
    width: int = 640
    height: int = 480
    fps: int = 15


@dataclass(frozen=True)
class RobotConfig:
    type: str = "realman"
    ip: str = "192.168.1.18"
    port: int = 8080


@dataclass(frozen=True)
class BoardConfig:
    """ChArUco board geometry. Lengths are in metres."""

    type: str = "charuco"
    squares_x: int = 14
    squares_y: int = 9
    square_length: float = 0.020
    marker_length: float = 0.015
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False


@dataclass(frozen=True)
class CollectorConfig:
    frame_timeout_ms: int = 200
    max_frame_age: float = 0.5
    min_corners: int = 30
    max_rms: float = 1.0
    window_name: str = "HandEye Collection"
    panel_height: int = 100
    output_dir: str = "handeye_data"


@dataclass(frozen=True)
class HandEyeConfig:
    camera: CameraConfig
    robot: RobotConfig
    board: BoardConfig
    collector: CollectorConfig
    # Absolute path of the directory that session folders are created in.
    output_root: Path
    # The file this config was loaded from, or None if it was built in code.
    config_path: Path | None = None


def _build_section(name: str, cls: type, raw: Any) -> Any:
    """Instantiate one config dataclass from a YAML mapping.

    Unknown keys raise rather than being ignored: a mistyped `min_corner:`
    would otherwise leave `min_corners` at its default and produce a run that
    looks entirely normal.
    """
    if raw is None:
        raise ValueError(f"Missing config section: {name!r}")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config section {name!r} must be a mapping, got {type(raw).__name__}"
        )

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in config section {name!r}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    return cls(**raw)


def load_config(path: str | Path | None = None) -> HandEyeConfig:
    """Read a YAML config file and return it as validated dataclasses.

    `yaml` is imported here rather than at module scope so that the config
    dataclasses stay importable on a machine without PyYAML.
    """
    import yaml

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(
            f"Unknown section(s) in {config_path.name}: {', '.join(unknown)}. "
            f"Valid sections: {', '.join(_SECTIONS)}"
        )

    collector = _build_section("collector", CollectorConfig, raw.get("collector"))

    output_root = Path(collector.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (PACKAGE_ROOT / output_root).resolve()

    return HandEyeConfig(
        camera=_build_section("camera", CameraConfig, raw.get("camera")),
        robot=_build_section("robot", RobotConfig, raw.get("robot")),
        board=_build_section("board", BoardConfig, raw.get("board")),
        collector=collector,
        output_root=output_root,
        config_path=config_path,
    )
