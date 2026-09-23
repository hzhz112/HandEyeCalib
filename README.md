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
| `solve_handeye.py` | Solver. Reads a session's `samples.json`. No OpenCV, no hardware. |
| `collect_handeye_data.py` | The pre-refactor monolith. Kept only as a comparison oracle; unmaintained. |

The dependency direction is one-way: `apps/` -> `board/`, `dataset/`, `camera/`,
`robot/` -> `config/`. Nothing under `camera/` or `robot/` may contain ChArUco
detection, `solvePnP`, data saving, JSON handling or OpenCV window code.

## Running

Needs the RealMan SDK, OpenCV with aruco, the RealSense SDK, NumPy, SciPy and
PyYAML.

```bash
python apps/collect_handeye.py               # uses config/handeye.yaml
python apps/collect_handeye.py --config PATH
```

Run it from the repository root, or from anywhere at all.
`apps/collect_handeye.py` prepends the repository root to `sys.path` when it is
executed as a script, so the sibling packages import regardless of the working
directory.

Imports inside this repository are rooted **at this directory** - `from board...`,
`from config...`, `from tests...`. Nothing depends on the checkout directory
being named `calibration/`, and neither `-m` nor a parent on `sys.path` is
required.

Press `S` to save one sample, `Q` or `ESC` to quit. The arm must be stationary -
the robot pose and the image are read one after the other, not synchronously.
Data lands in `handeye_data/session_<timestamp>/`.

## Solving

`solve_handeye.py` is a **standalone script**, unlike the collector: it imports
nothing else in this repository, so it runs directly from anywhere - no `-m`,
nothing to put on `sys.path`.

```bash
python solve_handeye.py                       # newest handeye_data/session_*/
python solve_handeye.py handeye_data/session_20260923_132552
python solve_handeye.py handeye_data/session_20260923_132552/samples.json
```

With no argument it picks the newest `handeye_data/session_*/samples.json`; a
directory argument resolves to the `samples.json` inside it.

It needs only NumPy and SciPy - **not** OpenCV, not the SDKs, not the hardware.
It solves `^F T_C` (color camera -> robot flange), holds out the last 5 samples
for validation, prints a consistency check on the training set and on the
held-out samples, and writes `handeye_result.json` next to the input
`samples.json`.

The solver shares nothing with `collect_handeye.py` except the `samples.json`
schema. It reads only `samples`, `index`, `T_base_flange`, `T_color_board`,
`charuco_corners` and `reprojection_rms_px`.

`handeye_result.json` is deliberately small: the transform, the camera origin,
the sample counts, and the aggregate errors. It records `method` (`"PARK"`) and
`solver` (`"park_numpy"`) so a result carries both the algorithm and the
implementation that produced it.

The transform is written the way a human writes a matrix - one line per row -
rather than the one-number-per-line `json.dump` default. The per-pose error
breakdown is not persisted: a calibration is judged by its mean and worst-case
error, and a 15-entry array of residuals is not something anyone reads. The
console output carries the same information.

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

Or one module at a time, from this directory:

```bash
python -m tests.test_charuco   # no hardware: renders a board and diffs
python -m tests.test_dataset   # no hardware: samples.json contract
python -m tests.test_camera    # needs the D435; skips cleanly without it
python -m tests.test_robot     # skips in ~1 s if the arm is unreachable
```

Tests skip rather than fail when a device is absent. `pytest.ini` sets
`pythonpath = .` (imports are rooted at this directory) and restricts
collection to `tests/`: `test/` holds interactive
scripts that open a camera and loop forever at import time, and
`tests/test_show_information.py` connects to the arm at import time.

`tests/test_charuco.py` diffs the detector against `collect_handeye_data.py`,
the pre-refactor implementation. Keep that file until the comparison is no
longer useful.

## Hand-eye solve: Park's method implemented in NumPy

`solve_handeye.py` does **not** call `cv2.calibrateHandEye`. OpenCV 5.0 removed
that function: in 5.0.0.93, `hasattr(cv2, "calibrateHandEye")` is `False` at top
level, in every submodule and in the shipped stubs, while `cv2.CALIB_HAND_EYE_PARK`
still exists - so the failure appeared at the call site, not at the import.

Downgrading OpenCV was the alternative, and was rejected: the aruco API differs
between 4.x and 5.0, so the collector would have needed re-validating, and the
collected data was produced under 5.0. Instead the package implements the
algorithm itself and drops the OpenCV dependency from the solver entirely.

Park & Martin (1994), "Robot sensor calibration: solving AX = XB on the Euclidean
group", is a **closed-form** solution - no iteration, no initial guess. The
rotation part of `A X = X B` gives `alpha_ij = R_X beta_ij` for the relative
motion rotation vectors, which is linear in `R_X` and solves by orthogonal
Procrustes (SVD). The translation then follows from a least-squares solve of
`(R_A - I) t_X = R_X t_B - t_A` over the same pose pairs. About 40 lines of NumPy;
see `solve_handeye()` for the derivation.

**Verified against OpenCV 4.14.0.94's own `CALIB_HAND_EYE_PARK`**, on the
15-sample training set of `session_20260923_132552`:

| Quantity | Agreement with OpenCV PARK |
|---|---|
| Rotation matrix, max element | 3.0e-14 |
| Rotation, degrees | 1.3e-13 |
| Translation, mm | 1.2e-11 |

That is floating-point rounding, not algorithmic difference. For context, the
other OpenCV methods differ from PARK by up to 1.8 mm on the same data, so the
reproduction is exact for the method the result is labelled with.

The reason the solver validates the *convention* as well as the numbers is that a
transposed or inverted `T_flange_color` produces a finite, orthonormal, entirely
plausible - and wrong - matrix. `evaluate()` catches that independently of the
solver, by checking that every pose places the board at the same base-frame pose.

Park's method needs relative rotations about **more than one axis**; about a
single axis, the rotation about that axis is unobservable. Both the rotation
diversity check in `main()` and the rank check in `solve_handeye()` warn about
this, because the failure mode is otherwise silent - the method returns a
well-formed, wrong answer.
