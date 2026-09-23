
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ==================== 配置 ====================

ROOT = Path(__file__).resolve().parent / "handeye_data"
MIN_CORNERS = 30
MAX_RMS = 1.0
TEST_COUNT = 5                 # 预留最后 5 组做独立验证
MIN_TRAIN = 8
METHOD = cv2.CALIB_HAND_EYE_PARK


# ==================== 工具函数 ====================

def load_data():
    """读取命令行指定的 JSON，或自动读取最新的采集文件。"""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if path.is_dir():
            path = path / "samples.json"
    else:
        files = sorted(ROOT.glob("session_*/samples.json"))
        if not files:
            raise FileNotFoundError(f"未找到采集数据，请检查目录：{ROOT}")
        path = files[-1]

    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"读取文件：{path}")
    return data, path


def check_matrix(value, name):
    """检查齐次变换矩阵的尺寸、数值及旋转矩阵有效性。"""
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} 不是有效的 4x4 矩阵")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} 的最后一行不正确")

    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1, atol=1e-3):
        raise ValueError(f"{name} 的旋转矩阵无效")
    return T


def make_matrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return T


def rotation_error(R1, R2):
    """两个旋转矩阵之间的夹角，单位：度。"""
    return float(np.degrees(Rotation.from_matrix(R1.T @ R2).magnitude()))


def mean_transform(transforms):
    """计算一组位姿的平均平移与平均旋转，仅用于验证。"""
    positions = np.array([T[:3, 3] for T in transforms])
    rotations = Rotation.from_matrix(np.array([T[:3, :3] for T in transforms]))
    return make_matrix(rotations.mean().as_matrix(), positions.mean(axis=0))


# ==================== 手眼标定 ====================

def solve_handeye(samples):
    """输入采集样本，求解 ^F T_C（彩色相机 -> 法兰）。"""
    flange_poses = [s["T_base_flange"] for s in samples]
    board_poses = [s["T_color_board"] for s in samples]

    R_g2b = [T[:3, :3] for T in flange_poses]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in flange_poses]
    R_t2c = [T[:3, :3] for T in board_poses]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in board_poses]

    R_c2f, t_c2f = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=METHOD
    )
    T_flange_color = make_matrix(R_c2f, t_c2f)

    if not np.isfinite(T_flange_color).all():
        raise RuntimeError("手眼标定结果包含 NaN 或 Inf，请检查采集姿态")

    return check_matrix(T_flange_color, "T_flange_color")


# ==================== 标定验证 ====================

def evaluate(samples, T_flange_color, reference, name):
    """检查不同姿态算出的标定板基座位姿是否一致。"""
    position_errors, angle_errors, details = [], [], []

    for sample in samples:
        T_base_board = sample["T_base_flange"] @ T_flange_color @ sample["T_color_board"]
        pos_mm = float(np.linalg.norm(T_base_board[:3, 3] - reference[:3, 3]) * 1000)
        angle_deg = rotation_error(reference[:3, :3], T_base_board[:3, :3])

        position_errors.append(pos_mm)
        angle_errors.append(angle_deg)
        details.append({
            "index": sample["index"],
            "position_error_mm": pos_mm,
            "rotation_error_deg": angle_deg
        })

    p, a = np.asarray(position_errors), np.asarray(angle_errors)
    print(f"\n========== {name}：{len(samples)} 组 ==========")
    print(f"位置误差：平均 {p.mean():.3f} mm | 最大 {p.max():.3f} mm")
    print(f"旋转误差：平均 {a.mean():.3f} deg | 最大 {a.max():.3f} deg")

    return {
        "count": len(samples),
        "mean_position_error_mm": float(p.mean()),
        "max_position_error_mm": float(p.max()),
        "mean_rotation_error_deg": float(a.mean()),
        "max_rotation_error_deg": float(a.max()),
        "samples": details
    }


# ==================== 主程序 ====================

def main():
    data, input_path = load_data()
    raw_samples = data.get("samples", [])
    samples = []

    for s in raw_samples:
        try:
            if s.get("charuco_corners", 0) < MIN_CORNERS:
                raise ValueError("角点数量不足")
            if float(s.get("reprojection_rms_px", np.inf)) > MAX_RMS:
                raise ValueError("重投影误差过大")

            item = {
                "index": s["index"],
                "T_base_flange": check_matrix(s["T_base_flange"], "T_base_flange"),
                "T_color_board": check_matrix(s["T_color_board"], "T_color_board")
            }
            samples.append(item)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[跳过] 第 {s.get('index', '?')} 组：{e}")

    print(f"\n原始数据：{len(raw_samples)} 组 | 有效数据：{len(samples)} 组")
    if len(samples) < MIN_TRAIN:
        raise RuntimeError(f"有效数据不足，至少需要 {MIN_TRAIN} 组")

    # 数据足够时预留最后 5 组验证，否则全部用于求解
    n_test = TEST_COUNT if len(samples) >= MIN_TRAIN + TEST_COUNT else 0
    train = samples[:-n_test] if n_test else samples
    test = samples[-n_test:] if n_test else []

    # 提醒检查旋转姿态的多样性
    rotations = [s["T_base_flange"][:3, :3] for s in train]
    max_rotation = max(
        rotation_error(R1, R2)
        for i, R1 in enumerate(rotations)
        for R2 in rotations[i + 1:]
    )
    print(f"训练集最大法兰相对旋转角：{max_rotation:.2f} deg")
    if max_rotation < 15:
        print("[提醒] 法兰旋转变化偏小，建议补充不同方向的旋转姿态")

    # 计算 ^F T_C
    T_flange_color = solve_handeye(train)

    print("\n========== 手眼标定结果 ==========")
    print("T_flange_color（彩色相机 -> 法兰）：")
    print(np.array2string(T_flange_color, precision=6, suppress_small=True))
    print("\n相机原点在法兰坐标系下的位置（mm）：")
    print(np.round(T_flange_color[:3, 3] * 1000, 3))

    # 使用训练集求出标定板在机器人基座下的平均位姿
    train_board_poses = [
        s["T_base_flange"] @ T_flange_color @ s["T_color_board"]
        for s in train
    ]
    T_base_board_ref = mean_transform(train_board_poses)

    train_result = evaluate(train, T_flange_color, T_base_board_ref, "训练集一致性")
    test_result = evaluate(test, T_flange_color, T_base_board_ref, "预留数据验证") if test else None

    result = {
        "source_file": str(input_path),
        "method": "PARK",
        "coordinate_convention": {
            "T_flange_color": "color_camera_to_robot_flange",
            "T_base_board_reference": "board_to_robot_base",
            "translation_unit": "m",
            "rotation_matrix": "3x3"
        },
        "raw_sample_count": len(raw_samples),
        "valid_sample_count": len(samples),
        "training_sample_indices": [s["index"] for s in train],
        "validation_sample_indices": [s["index"] for s in test],
        "T_flange_color": T_flange_color.tolist(),
        "T_base_board_reference": T_base_board_ref.tolist(),
        "training_metrics": train_result,
        "validation_metrics": test_result
    }

    output_path = input_path.parent / "handeye_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n标定结果已保存：{output_path}")


if __name__ == "__main__":
    main()