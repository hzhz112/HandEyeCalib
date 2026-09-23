
import os
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from calibration.test.test_realman_pose import get_realman_pose

if os.name == "nt":
    import msvcrt


# ==================== 配置 ====================

ROBOT_IP, ROBOT_PORT = "192.168.1.18", 8080
WIDTH, HEIGHT, FPS = 640, 480, 15

SQUARES_X, SQUARES_Y = 14, 9
SQUARE_LENGTH, MARKER_LENGTH = 0.020, 0.015

MIN_CORNERS = 30
MAX_RMS = 1.0
MAX_FRAME_AGE = 0.5

WINDOW_NAME = "RealMan D435 HandEye Collection"
SAVE_DIR = Path(__file__).resolve().parent / "handeye_data" / datetime.now().strftime("session_%Y%m%d_%H%M%S")
IMAGE_DIR = SAVE_DIR / "images"
JSON_PATH = SAVE_DIR / "samples.json"


# ==================== 标定板 ====================

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
board.setLegacyPattern(False)

charuco_detector = cv2.aruco.CharucoDetector(board)
board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)


# ==================== 工具函数 ====================

def get_key():
    key = cv2.waitKeyEx(10)
    if key != -1:
        return key
    if os.name == "nt" and msvcrt.kbhit():
        return ord(msvcrt.getwch().lower())
    return -1


def pose_to_matrix(pose):
    """RealMan [x, y, z, rx, ry, rz] -> 法兰到基座的 4×4 矩阵。"""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size != 6 or not np.isfinite(pose).all():
        raise ValueError(f"无效的 RealMan 末端位姿: {pose}")

    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=False).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def prepare_image_points(points, intr, K):
    """处理 D435 彩色相机的畸变模型。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if intr.model == rs.distortion.none:
        return points, np.zeros(5)

    if intr.model == rs.distortion.brown_conrady:
        return points, np.asarray(intr.coeffs, dtype=np.float64)

    if intr.model == rs.distortion.inverse_brown_conrady:
        corrected = []
        for u, v in points:
            x, y, z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0)
            corrected.append([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]])
        return np.asarray(corrected, dtype=np.float64), np.zeros(5)

    raise RuntimeError(f"暂不支持的畸变模型: {intr.model}")


def detect_board(image, intr, K):
    """检测 ChArUco，并返回标定板到彩色相机的 4×4 矩阵。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    marker_count = 0 if marker_ids is None else len(marker_ids)
    corner_count = 0 if ids is None else len(np.asarray(ids).reshape(-1))
    result = {"markers": marker_count, "corners": corner_count, "rms": None, "T": None, "points": []}

    if corners is None or ids is None:
        return result

    points = np.asarray(corners, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)

    if points.size % 2:
        return result

    points = points.reshape(-1, 2)
    if len(points) != len(ids):
        return result

    result["points"] = points
    if len(ids) < MIN_CORNERS or np.any(ids < 0) or np.any(ids >= len(board_points)):
        return result
    if len(np.unique(ids)) != len(ids):
        return result

    object_points = np.ascontiguousarray(board_points[ids], dtype=np.float64)
    image_points, D = prepare_image_points(points, intr, K)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or float(tvec[2, 0]) <= 0:
        return result

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    error = projected.reshape(-1, 2) - image_points
    rms = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec.reshape(3)

    result.update({"rms": rms, "T": T})
    return result


def save_sample(arm, latest, samples, camera_info):
    """按 S 保存一组相机与机器人配对数据。机械臂必须处于静止状态。"""
    if latest is None:
        print("[跳过] 尚未收到相机图像")
        return

    age = time.monotonic() - latest["monotonic_time"]
    if age > MAX_FRAME_AGE:
        print(f"[跳过] 图像已过期: {age:.2f} s")
        return

    detection = latest["detection"]
    if detection["T"] is None:
        print("[跳过] 当前没有有效的标定板位姿")
        return
    if detection["rms"] > MAX_RMS:
        print(f"[跳过] 重投影误差过大: {detection['rms']:.3f} px")
        return

    robot_pose = np.asarray(get_realman_pose(arm), dtype=np.float64).reshape(-1)
    robot_time = time.time()
    T_base_flange = pose_to_matrix(robot_pose)

    index = len(samples) + 1
    image_name = f"sample_{index:03d}.png"
    image_path = IMAGE_DIR / image_name

    if not cv2.imwrite(str(image_path), latest["image"]):
        raise RuntimeError(f"图像保存失败: {image_path}")

    sample = {
        "index": index,
        "camera_host_time": latest["host_time"],
        "robot_host_time": robot_time,
        "realsense_timestamp_ms": latest["rs_timestamp_ms"],
        "robot_pose_xyz_rpy": robot_pose.tolist(),
        "T_base_flange": T_base_flange.tolist(),
        "T_color_board": detection["T"].tolist(),
        "aruco_markers": detection["markers"],
        "charuco_corners": detection["corners"],
        "reprojection_rms_px": detection["rms"],
        "image": f"images/{image_name}"
    }

    samples.append(sample)
    data = {
        "camera": camera_info,
        "board": {
            "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_LENGTH, "marker_length_m": MARKER_LENGTH,
            "dictionary": "DICT_5X5_100", "legacy_pattern": False
        },
        "coordinate_convention": {
            "T_base_flange": "flange_to_robot_base",
            "T_color_board": "board_to_color_camera",
            "position_unit": "m", "euler_unit": "rad",
            "rotation": "Rz(rz) @ Ry(ry) @ Rx(rx)"
        },
        "samples": samples
    }

    temp_path = JSON_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, JSON_PATH)

    print(f"[保存成功] 第 {index} 组 | 角点: {detection['corners']} | RMS: {detection['rms']:.3f} px")
    print(f"法兰位姿: {robot_pose.tolist()}")
    print(f"数据文件: {JSON_PATH}\n")


# ==================== 主程序 ====================

def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    robot_connected = camera_started = False
    samples, latest = [], None

    try:
        print(f"正在连接 RealMan: {ROBOT_IP}:{ROBOT_PORT}")
        handle = arm.rm_create_robot_arm(ROBOT_IP, ROBOT_PORT)
        if handle is None or handle.id <= 0:
            raise RuntimeError("RealMan 连接失败")

        robot_connected = True
        print("RealMan 连接成功，当前法兰位姿:", get_realman_pose(arm))

        profile = pipeline.start(config)
        camera_started = True

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

        supported = (rs.distortion.none, rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady)
        if intr.model not in supported:
            raise RuntimeError(f"暂不支持的相机畸变模型: {intr.model}")

        camera_info = {
            "stream": "color", "width": intr.width, "height": intr.height, "fps": FPS,
            "K": K.tolist(), "distortion_model": str(intr.model),
            "distortion_coefficients": list(intr.coeffs)
        }

        print(f"D435 启动成功 | 畸变模型: {intr.model}")
        print("S: 保存数据 | Q / ESC: 退出")
        print("请在机械臂停止且画面稳定后按 S")
        print("保存目录:", SAVE_DIR)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
            except RuntimeError:
                color_frame = None

            if color_frame:
                image = np.asanyarray(color_frame.get_data()).copy()
                frame_time = time.monotonic()
                detection = detect_board(image, intr, K)

                latest = {
                    "image": image, "detection": detection,
                    "host_time": time.time(), "monotonic_time": frame_time,
                    "rs_timestamp_ms": float(color_frame.get_timestamp())
                }

                display = image.copy()
                for u, v in detection["points"]:
                    cv2.circle(display, (int(round(u)), int(round(v))), 2, (0, 255, 255), -1)

                panel = np.zeros((100, WIDTH, 3), dtype=np.uint8)
                text = f"ArUco: {detection['markers']}/63   ChArUco: {detection['corners']}/104"
                cv2.putText(panel, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                if detection["T"] is not None:
                    x, y, z = detection["T"][:3, 3] * 1000
                    text = f"X:{x:+.1f}  Y:{y:+.1f}  Z:{z:+.1f} mm"
                    cv2.putText(panel, text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    status = "READY" if detection["rms"] <= MAX_RMS else "RMS TOO HIGH"
                    text = f"RMS:{detection['rms']:.3f}px   {status}   Saved:{len(samples)}"
                else:
                    text = f"Pose unavailable   Saved:{len(samples)}"

                cv2.putText(panel, text, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(WINDOW_NAME, np.vstack((display, panel)))

            key = get_key()
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                try:
                    save_sample(arm, latest, samples, camera_info)
                except Exception as e:
                    print("[保存失败]", repr(e))

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出")

    finally:
        if camera_started:
            pipeline.stop()
        if robot_connected:
            arm.rm_delete_robot_arm()
        cv2.destroyAllWindows()
        print(f"程序结束，本次成功采集 {len(samples)} 组")
        if samples:
            print("数据文件:", JSON_PATH)


if __name__ == "__main__":
    main()