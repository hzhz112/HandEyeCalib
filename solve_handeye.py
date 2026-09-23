
"""Hand-eye calibration solver.

Reads a collection session's `samples.json`, solves `^F T_C` (color camera ->
robot flange), validates the result against held-out samples, and writes
`handeye_result.json` next to the input file.

This module has no OpenCV dependency. OpenCV 5.0 removed `cv2.calibrateHandEye`,
and Park's method - the algorithm behind `CALIB_HAND_EYE_PARK` - is a closed-form
sequence of linear algebra that does not need it. See `solve_handeye()`.

Run from this directory:

    python solve_handeye.py                # newest handeye_data/session_*/
    python solve_handeye.py SESSION_DIR    # a session directory
    python solve_handeye.py samples.json   # a samples.json file
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# ==================== Configuration ====================

ROOT = Path(__file__).resolve().parent / "handeye_data"
MIN_CORNERS = 30
MAX_RMS = 1.0
TEST_COUNT = 5                 # hold out the last 5 samples for validation
MIN_TRAIN = 8
METHOD = "PARK"


# ==================== Helpers ====================

def load_data():
    """Read the JSON named on the command line, or the newest collection."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if path.is_dir():
            path = path / "samples.json"
    else:
        files = sorted(ROOT.glob("session_*/samples.json"))
        if not files:
            raise FileNotFoundError(f"No collected data; checked: {ROOT}")
        path = files[-1]

    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Reading: {path}")
    return data, path


def check_matrix(value, name):
    """Validate a homogeneous transform: shape, finiteness, rotation validity."""
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} is not a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} has an invalid last row")

    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1, atol=1e-3):
        raise ValueError(f"{name} has an invalid rotation matrix")
    return T


def make_matrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return T


def rotation_error(R1, R2):
    """Angle between two rotation matrices, in degrees."""
    return float(np.degrees(Rotation.from_matrix(R1.T @ R2).magnitude()))


def mean_transform(transforms):
    """Mean translation and mean rotation of a set of poses. Validation only."""
    positions = np.array([T[:3, 3] for T in transforms])
    rotations = Rotation.from_matrix(np.array([T[:3, :3] for T in transforms]))
    return make_matrix(rotations.mean().as_matrix(), positions.mean(axis=0))


# `json.dump` puts every number on its own line, which spreads a 4x4 transform
# over 16 of them and a 3-vector over 3. These placeholders are dumped in place
# of those values and swapped for pre-rendered text afterwards, so everything
# else keeps json's escaping and number formatting. No real string in this file
# can collide with them.
MATRIX_SLOT = "@@matrix@@"
VECTOR_SLOT = "@@vector@@"


def format_row(values, decimals=6):
    """Render numbers as a single-line JSON array."""
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def format_matrix(T, decimals=6, indent=2):
    """Render a matrix the way a human writes it: one line per row.

    `indent` is the indent of the key this matrix is the value of; the rows are
    placed one level deeper, and the closing bracket lines up with the key.
    """
    pad = " " * indent
    rows = [f"{pad}  {format_row(row, decimals)}" for row in T]
    return "[\n" + ",\n".join(rows) + f"\n{pad}]"


# ==================== Hand-eye solve ====================

def project_to_so3(M):
    """Nearest rotation matrix to M in Frobenius norm.

    Orthogonal Procrustes via SVD: the optimum is `U V^T`, with the last
    singular value negated when `U V^T` is a reflection instead of a rotation.
    """
    U, _, Vt = np.linalg.svd(M)
    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[2, 2] = -1.0
    return U @ correction @ Vt


def pair_motions(flange_poses, board_poses):
    """Relative motions of every pose pair, in the form `A_ij X = X B_ij`.

    Poses i and j observe the same, fixed board, so

        T_flange_i @ X @ T_board_i == T_flange_j @ X @ T_board_j

    with `X = ^F T_C`. Rearranging gives `A_ij X = X B_ij` where

        A_ij = inv(T_flange_j) @ T_flange_i    flange motion, in the base frame
        B_ij = T_board_j @ inv(T_board_i)      board motion, in the camera frame

    Both sides depend only on measured poses; `X` is the unknown.
    """
    pairs = []
    for i in range(len(flange_poses)):
        for j in range(i + 1, len(flange_poses)):
            A = np.linalg.inv(flange_poses[j]) @ flange_poses[i]
            B = board_poses[j] @ np.linalg.inv(board_poses[i])
            pairs.append((A, B))
    return pairs


def solve_handeye(samples):
    """Solve `^F T_C` (color camera -> robot flange) with Park's method.

    Park & Martin (1994), "Robot sensor calibration: solving AX = XB on the
    Euclidean group". The rotation part of `A X = X B` obeys

        R_A R_X = R_X R_B

    so `R_A` is `R_B` conjugated by `R_X`: the two share a rotation angle and
    their axes are related by `alpha_ij = R_X beta_ij`, where `alpha_ij` and
    `beta_ij` are the rotation vectors (axis * angle) of `R_A` and `R_B`. That
    relation is linear in `R_X`, so stacking it over every pose pair and solving
    by orthogonal Procrustes recovers `R_X` in closed form - no iteration and no
    initial guess, unlike `CALIB_HAND_EYE_TSAI`'s nonlinear refinement.

    The translation then follows from the translation part of `A X = X B`,

        (R_A - I) t_X = R_X t_B - t_A

    stacked over the same pairs and solved in the least-squares sense.
    """
    flange_poses = [s["T_base_flange"] for s in samples]
    board_poses = [s["T_color_board"] for s in samples]
    pairs = pair_motions(flange_poses, board_poses)

    alpha = np.column_stack([Rotation.from_matrix(A[:3, :3]).as_rotvec() for A, _ in pairs])
    beta = np.column_stack([Rotation.from_matrix(B[:3, :3]).as_rotvec() for _, B in pairs])

    # Relative rotations about a single axis leave the rotation about that axis
    # unobservable, and Park's method would return a confident wrong answer.
    rank = np.linalg.matrix_rank(beta, tol=1e-8)
    if rank < 3:
        print(f"[Warning] Relative rotations span only {rank} axis/axes, so the")
        print("          hand-eye rotation is not fully observable. Collect poses")
        print("          that rotate about more than one axis.")

    # `R_X` is the rotation mapping every column of `beta` onto its `alpha`.
    R_c2f = project_to_so3(alpha @ beta.T)

    design = np.vstack([A[:3, :3] - np.eye(3) for A, _ in pairs])
    target = np.concatenate([R_c2f @ B[:3, 3] - A[:3, 3] for A, B in pairs])
    t_c2f = np.linalg.lstsq(design, target, rcond=None)[0]

    T_flange_color = make_matrix(R_c2f, t_c2f)

    if not np.isfinite(T_flange_color).all():
        raise RuntimeError("Hand-eye result contains NaN or Inf; check the collected poses")

    return check_matrix(T_flange_color, "T_flange_color")


# ==================== Calibration validation ====================

def evaluate(samples, T_flange_color, reference, name):
    """Check that every pose places the board at the same base-frame pose.

    Independent of the solver: a wrong result, or an inverted coordinate
    convention, shows up here as errors orders of magnitude above noise.

    Returns the aggregate errors only. A calibration is judged by these numbers,
    so the per-pose breakdown is not persisted.
    """
    position_errors, angle_errors = [], []

    for sample in samples:
        T_base_board = sample["T_base_flange"] @ T_flange_color @ sample["T_color_board"]
        position_errors.append(
            float(np.linalg.norm(T_base_board[:3, 3] - reference[:3, 3]) * 1000)
        )
        angle_errors.append(rotation_error(reference[:3, :3], T_base_board[:3, :3]))

    p, a = np.asarray(position_errors), np.asarray(angle_errors)
    print(f"\n========== {name}: {len(samples)} samples ==========")
    print(f"Position error: mean {p.mean():.3f} mm | max {p.max():.3f} mm")
    print(f"Rotation error: mean {a.mean():.3f} deg | max {a.max():.3f} deg")

    # Three decimals - a micrometre and a milli-degree - is well past the
    # repeatability of the measurements themselves. Full float precision here
    # would be noise dressed up as accuracy.
    return {
        "position_error_mm": {"mean": round(float(p.mean()), 3), "max": round(float(p.max()), 3)},
        "rotation_error_deg": {"mean": round(float(a.mean()), 3), "max": round(float(a.max()), 3)},
    }


# ==================== Main ====================

def main():
    data, input_path = load_data()
    raw_samples = data.get("samples", [])
    samples = []

    for s in raw_samples:
        try:
            if s.get("charuco_corners", 0) < MIN_CORNERS:
                raise ValueError("too few corners")
            if float(s.get("reprojection_rms_px", np.inf)) > MAX_RMS:
                raise ValueError("reprojection error too large")

            item = {
                "index": s["index"],
                "T_base_flange": check_matrix(s["T_base_flange"], "T_base_flange"),
                "T_color_board": check_matrix(s["T_color_board"], "T_color_board")
            }
            samples.append(item)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[Skipped] sample {s.get('index', '?')}: {e}")

    print(f"\nRaw samples: {len(raw_samples)} | usable: {len(samples)}")
    if len(samples) < MIN_TRAIN:
        raise RuntimeError(f"Not enough usable samples; need at least {MIN_TRAIN}")

    # Hold out the last TEST_COUNT samples when there are enough to spare.
    n_test = TEST_COUNT if len(samples) >= MIN_TRAIN + TEST_COUNT else 0
    train = samples[:-n_test] if n_test else samples
    test = samples[-n_test:] if n_test else []

    # Rotation diversity: Park's method needs relative rotations about more than
    # one axis, or the hand-eye rotation is under-determined.
    rotations = [s["T_base_flange"][:3, :3] for s in train]
    max_rotation = max(
        rotation_error(R1, R2)
        for i, R1 in enumerate(rotations)
        for R2 in rotations[i + 1:]
    )
    print(f"Max relative flange rotation in training set: {max_rotation:.2f} deg")
    if max_rotation < 15:
        print("[Warning] Flange rotation is small; add poses with more varied rotation")

    # Compute ^F T_C
    T_flange_color = solve_handeye(train)

    print("\n========== Hand-eye result ==========")
    print("T_flange_color (color camera -> robot flange):")
    print(np.array2string(T_flange_color, precision=6, suppress_small=True))
    print("\nCamera origin in the flange frame (mm):")
    print(np.round(T_flange_color[:3, 3] * 1000, 3))

    # Average board pose in the robot base frame, from the training set.
    train_board_poses = [
        s["T_base_flange"] @ T_flange_color @ s["T_color_board"]
        for s in train
    ]
    T_base_board_ref = mean_transform(train_board_poses)

    train_result = evaluate(train, T_flange_color, T_base_board_ref, "Training set consistency")
    test_result = evaluate(test, T_flange_color, T_base_board_ref, "Held-out validation") if test else None

    result = {
        "source_file": str(input_path),
        "method": METHOD,
        "solver": "park_numpy",
        "convention": "T_flange_color maps color camera -> robot flange; translation in m",
        "samples": {
            "total": len(raw_samples),
            "used": len(samples),
            "training": len(train),
            "validation": len(test),
        },
        "T_flange_color": MATRIX_SLOT,
        "camera_origin_in_flange_mm": VECTOR_SLOT,
        "training_error": train_result,
        "validation_error": test_result,
    }

    output_path = input_path.parent / "handeye_result.json"
    text = json.dumps(result, ensure_ascii=False, indent=2)
    for slot, rendered in (
        (MATRIX_SLOT, format_matrix(T_flange_color)),
        (VECTOR_SLOT, format_row(T_flange_color[:3, 3] * 1000, decimals=3)),
    ):
        text = text.replace(f'"{slot}"', rendered)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nResult written to: {output_path}")


if __name__ == "__main__":
    main()
