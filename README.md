# Handeye calibration: data collection and solve

Collects paired camera/robot observations of a ChArUco board and solves the
handeye transform. The package is layered so that a vendor SDK only ever
appears inside the adapter that owns it.

## Layout

| Path | Responsibility |
|---|---|
| `camera/` | Device layer. Camera adapters, their interface, and the factory. |
| `robot/` | Device layer. Robot adapters, their interface, and the factory. |
| `board/charuco_detector.py` | Domain layer. Board construction, corner detection, `solvePnP`, reprojection error. |
| `dataset/` | Domain layer. `HandEyeSample` and `DatasetWriter` (the only code that touches `cv2.imwrite` or `json.dump`). |
| `config/` | Leaf layer. `handeye.yaml` plus the typed loader every other layer depends on. |
| `apps/collect_handeye.py` | Application layer. Flow control only. |
| `tests/` | Test suite. `pytest.ini` at the repo root configures it. |
| `test/` | Interactive hardware scripts. Not tests; excluded from collection. |
| `solve_handeye.py` | Solver. Reads a session's `samples.json`. See the blocker at the end. |
| `collect_handeye_data.py` | The pre-refactor monolith. Kept only as a comparison oracle; unmaintained. |

The dependency direction is one-way: `apps/` -> `board/`, `dataset/`, `camera/`,
`robot/` -> `config/`. Nothing under `camera/` or `robot/` may contain ChArUco
detection, `solvePnP`, data saving, JSON handling or OpenCV window code.

## Running

Needs the RealMan SDK, OpenCV with aruco, the RealSense SDK, NumPy, SciPy and
PyYAML.

This repository directory **is** the `calibration` package directory, so run it
**from the directory that contains the repository**, as a module:

```bash
cd ..                                     # the parent of this repo
python -m calibration.apps.collect_handeye               # uses config/handeye.yaml
python -m calibration.apps.collect_handeye --config PATH
```

Clone the repo as a directory named `calibration/` and this works unchanged.

`python calibration/apps/collect_handeye.py` does **not** work: that puts the
script's own directory on `sys.path[0]`, so every `calibration.*` import fails.
Neither does running `python -m calibration.apps.collect_handeye` from *inside*
this directory - `calibration` is then not importable, because the parent is
what needs to be on the path. Only the `-m` form from the parent works.

Press `S` to save one sample, `Q` or `ESC` to quit. The arm must be stationary -
the robot pose and the image are read one after the other, not synchronously.
Data lands in `calibration/handeye_data/session_<timestamp>/`.

## Configuration

`config/handeye.yaml` is the single source of truth; the factories no longer
read environment variables. The loader rejects unknown keys, so a mistyped
`min_corner:` fails at startup instead of silently keeping the default.

`min_corners` and `max_rms` are also present in `solve_handeye.py`. The
collector accepting samples the solver will then discard is a silent failure,
so change both files together.

## Adding a camera

Write an adapter in `camera/`, subclass `CameraInterface`, implement the five
methods, then register it in `camera_factory.py` and set `camera.type` in the
YAML. `apps/` does not change.

Invariants an adapter must honour:

- `CameraFrame.image` is **BGR, uint8, shape `(H, W, 3)`**. Not RGB, not
  grayscale - the detector applies `COLOR_BGR2GRAY` to it and `cv2.imwrite`
  writes it as BGR.
- `CameraFrame.monotonic_time` comes from `time.monotonic()` and is **the only
  clock that may be used to judge whether a frame is stale**. `host_time` is
  wall clock and can jump under NTP.
- `CameraFrame.device_timestamp_ms` is the device clock. Different time base;
  never subtract it from `host_time`.
- Vendor objects stay inside the adapter. Do not leak an SDK's intrinsics
  object or distortion enum through the interface.
- `get_intrinsics()` returns `validate_intrinsics(...)`.
- Do not mutate `frame.image` in place. Copy before annotating.
- `stop()` must be safe to call twice, and before `start()`.

### Distortion

`prepare_image_points(points)` returns the `(image_points, dist_coeffs)` pair
that goes straight into `cv2.solvePnP(board_points, image_points, K, dist_coeffs)`
and `cv2.projectPoints(...)`, with `K` from `get_intrinsics()`.

The returned points are **not** the input points in general. `RealSenseD435`
overrides the method for `inverse_brown_conrady`, de-distorting point by point
through `rs.rs2_deproject_pixel_to_point` and returning `zeros(5)` as the
coefficients. Do **not** replace that loop with `cv2.undistortPoints`: it
cannot consume inverse Brown-Conrady coefficients.

Consequently, `BoardDetection.points` carries the **raw** corners for drawing,
which under this model are different coordinates from the ones `solvePnP`
consumed. Drawing the prepared points would misplace every overlay marker.

Two invariants that fail quietly if broken:

- The `K` used inside `prepare_image_points` must be the same `K` passed to
  `solvePnP`. `CharucoDetector` reads both from the same adapter for exactly
  this reason; never accept an intrinsics matrix from configuration. A mismatch
  yields a self-consistent, wrong `T_color_board` with a *deceptively low* RMS.
- The `charuco_corners` field is a **count**, not a point array.
  `solve_handeye.py` compares it numerically, and an array there raises a
  `ValueError` that the solver's broad `except` reports as "not enough corners"
  for every sample.

Note that the D435 in this repository reports all-zero distortion coefficients,
so its de-distortion path is numerically an identity map. That is a property of
this unit's lens, not of the code: do not write tests that assume de-distortion
moves the points.

## Adding a robot

Write an adapter in `robot/`, subclass `RobotInterface`, implement `connect`,
`get_base_T_flange` and `disconnect`, then register it in `robot_factory.py` and
set `robot.type` in the YAML.

`get_base_T_flange()` returns `^base T_flange` as a 4x4 with translation in
metres, so that `p_base = T_base_flange @ p_flange`. **If a vendor SDK hands
back the TCP rather than the flange, convert it inside the adapter**
(`T_base_flange = T_base_tcp @ inverse(T_flange_tcp)`) using the tool offset
read from the real robot. `connect()` and `disconnect()` must both be
idempotent.

## Coordinates and data format

`T_base_flange` is flange -> robot base. `T_color_board` is board -> color camera.

`samples.json` per sample: `index`, `camera_host_time`, `robot_host_time`,
`device_timestamp_ms`, `T_base_flange`, `T_color_board`, `aruco_markers`,
`charuco_corners`, `reprojection_rms_px`, `image`. Top level: `camera`, `robot`,
`board`, `coordinate_convention`, `samples`.

`solve_handeye.py` reads only `samples`, `index`, `T_base_flange`,
`T_color_board`, `charuco_corners` and `reprojection_rms_px`.

The internal attribute names in `HandEyeSample` differ from the JSON keys -
`T_camera_board` serializes to `T_color_board`, `reprojection_error` to
`reprojection_rms_px`, and so on. The mapping lives only in
`HandEyeSample.to_dict()`, so that this file and `solve_handeye.py` keep
working while the code reads better.

`camera.distortion_coefficients` is deliberately the **raw SDK coefficients**,
not the zeros used by `solvePnP` under `inverse_brown_conrady`. Confusing, but
it is the existing schema.

Camera and robot readings are not hardware-synchronized; the workflow requires
a stationary arm. Motion-while-calibrating needs a different design.

## Tests

```bash
python -m pytest                 # from inside this repo root; pythonpath=.. handles the rest
python -m pytest -q              # 16 passed, 1 skipped when only the arm is absent
```

Or one module at a time, from the dir that contains this repo:

```bash
python -m calibration.tests.test_charuco   # no hardware: renders a board and diffs
python -m calibration.tests.test_dataset   # no hardware: samples.json contract
python -m calibration.tests.test_camera    # needs the D435; skips cleanly without it
python -m calibration.tests.test_robot     # skips in ~1 s if the arm is unreachable
```

Tests skip rather than fail when a device is absent. `pytest.ini` sets
`pythonpath = ..` (this directory is the package, so its parent must be
importable) and restricts collection to `tests/`: `test/` holds interactive
scripts that open a camera and loop forever at import time, and
`tests/test_show_information.py` connects to the arm at import time.

`tests/test_charuco.py` diffs the detector against `collect_handeye_data.py`,
the pre-refactor implementation. Keep that file until the comparison is no
longer useful.

## Known blocker: OpenCV 5.0 removed `cv2.calibrateHandEye`

`solve_handeye.py` cannot run on the installed OpenCV 5.0.0.93.
`hasattr(cv2, "calibrateHandEye")` is `False` at top level, in every submodule
and in the shipped stubs - but `cv2.CALIB_HAND_EYE_PARK` still exists, so the
failure appears at the call, not at the import.

Two ways out:

1. Downgrade in the active environment: `pip install "opencv-contrib-python<5"`.
   Note the aruco API differs between 4.x and 5.0, so the collector would need
   re-validating afterwards.
2. Implement Park's method in NumPy (Kronecker product plus SVD on `AX = XB`)
   and drop the OpenCV call.

This is unrelated to the layering in this package and is not addressed here.
