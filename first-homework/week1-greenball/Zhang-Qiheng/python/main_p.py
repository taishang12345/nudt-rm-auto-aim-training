import cv2
import numpy as np
import os
import json
import csv
import math
from pathlib import Path
# ====================== 全局根路径：固定以当前脚本文件为基准，不受启动目录影响 ======================
# 获取当前main.py脚本完整绝对路径
SCRIPT_PATH = Path(__file__).resolve()
# 脚本所在文件夹 = 项目根目录
PROJECT_ROOT = SCRIPT_PATH.parent
# 固定资源子路径（全部相对于项目根目录）
CONFIG_FILE_PATH = PROJECT_ROOT / "config.json"
INPUT_VIDEO_DEFAULT = PROJECT_ROOT / "first-homework.mp4"
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "video"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data"

# ====================== 配置读取（自动基于项目根路径，任意位置运行都能找到config） ======================
def load_config():
    # 自动创建输出文件夹，不存在则生成
    OUTPUT_VIDEO_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DATA_DIR.mkdir(exist_ok=True, parents=True)

    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(f"配置文件缺失！请确认项目根目录存在 config.json\n完整路径：{CONFIG_FILE_PATH}")

    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    cfg = {}
    # 路径逻辑：如果json内路径是相对路径，自动拼接项目根；绝对路径直接使用
    def get_abs_path(json_rel_path):
        p = Path(json_rel_path)
        if p.is_absolute():
            return str(p)
        return str(PROJECT_ROOT / json_rel_path)

    # 路径
    cfg["input_video"] = get_abs_path(cfg_data["path"]["input_video"]) if "input_video" in cfg_data["path"] else str(INPUT_VIDEO_DEFAULT)
    cfg["output_video"] = get_abs_path(cfg_data["path"]["output_video"])
    cfg["output_data"] = get_abs_path(cfg_data["path"]["output_data"])
    # 相机参数
    cfg["camera_matrix"] = np.array(cfg_data["camera"]["camera_matrix"], dtype=np.float32)
    cfg["dist_coeffs"] = np.array(cfg_data["camera"]["dist_coeffs"], dtype=np.float32)
    cfg["ball_radius"] = cfg_data["camera"]["ball_radius"]
    # 颜色检测
    cfg["lower_green"] = np.array(cfg_data["color_detect"]["lower_green"], dtype=np.uint8)
    cfg["upper_green"] = np.array(cfg_data["color_detect"]["upper_green"], dtype=np.uint8)
    cfg["morph_kernel_size"] = cfg_data["color_detect"]["morph_kernel_size"]
    cfg["min_contour_area"] = cfg_data["color_detect"]["min_contour_area"]
    # 形状筛选
    cfg["min_center_std"] = cfg_data["shape_filter"]["min_center_std"]
    cfg["axis_ratio_weight"] = cfg_data["shape_filter"]["axis_ratio_weight"]
    cfg["center_bias_weight"] = cfg_data["shape_filter"]["center_bias_weight"]
    # 卡尔曼
    cfg["kf_process_noise"] = cfg_data["kalman"]["process_noise"]
    cfg["kf_measure_noise"] = cfg_data["kalman"]["measurement_noise"]
    cfg["max_lost_frame"] = cfg_data["kalman"]["max_lost_frame"]
    # 可视化
    cfg["speed_scale"] = cfg_data["visual"]["speed_scale"]
    cfg["show_track_line"] = cfg_data["visual"]["show_track_line"]
    cfg["track_line_length"] = cfg_data["visual"]["track_line_length"]
    # 有效区域
    cfg["margin_w_ratio"] = cfg_data["roi_margin"]["margin_w_ratio"]
    cfg["margin_h_ratio"] = cfg_data["roi_margin"]["margin_h_ratio"]
    # 新增常量
    cfg["border_margin_px"] = 8

    return cfg

# ====================== 卡尔曼滤波 ======================
class KalmanFilter2D:
    def __init__(self, dt, process_noise, measure_noise):
        self.dt = dt
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measure_noise
        self.P = np.eye(4, dtype=np.float32) * 1000

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, pt):
        z = np.array(pt, dtype=np.float32).reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].flatten()

# ====================== 新增工具函数 ======================
def is_near_border(rect, frame_w, frame_h, margin=8):
    """判断轮廓包围盒是否贴近画面边缘"""
    x, y, w, h = rect
    return (x <= margin or
            y <= margin or
            (x + w) >= frame_w - margin or
            (y + h) >= frame_h - margin)

def calc_contour_center_stdev(cnt, center):
    cx, cy = center
    distances = []
    for p in cnt:
        x, y = p[0]
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances.append(d)
    return np.std(distances)

def calc_speed_and_dir(prev_pt, curr_pt):
    """
    计算像素速度大小 + 运动方向角(°)
    :return: speed, angle_deg
    """
    if prev_pt is None or curr_pt is None:
        return 0.0, 0.0
    dx = curr_pt[0] - prev_pt[0]
    dy = curr_pt[1] - prev_pt[1]
    speed = math.hypot(dx, dy)
    rad = math.atan2(dy, dx)
    angle = math.degrees(rad)
    if angle < 0:
        angle += 360.0
    return round(speed, 2), round(angle, 2)

def get_min_angle_diff(a_deg, b_deg):
    """计算0~360°环形最小角度差，消除几百度虚假差值"""
    diff = abs(a_deg - b_deg)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff

# ====================== 目标检测（改造：贴边自适应阈值 + 时序择优惩罚） ======================
def auto_filter_green_objects(frame, cfg, pred_prev_center=None, pred_prev_speed=0.0):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg["lower_green"], cfg["upper_green"])

    kernel = np.ones((cfg["morph_kernel_size"], cfg["morph_kernel_size"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ball = None
    best_score = -1.0
    best_rect = (0,0,0,0)

    center_x, center_y = w // 2, h // 2
    border_margin = cfg["border_margin_px"]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["min_contour_area"]:
            continue

        rect = cv2.boundingRect(cnt)
        near_edge = is_near_border(rect, w, h, border_margin)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        try:
            ellipse = cv2.fitEllipse(cnt)
            (_, _), (major, minor), _ = ellipse
            if minor < 1e-3:
                continue
            axis_ratio = major / minor
        except:
            continue

        stdev = calc_contour_center_stdev(cnt, (cx, cy))
        # 贴边/中心两套阈值
        if near_edge:
            max_axis = 1.45
            max_std = cfg["min_center_std"] * 1.6
        else:
            max_axis = 1.20
            max_std = cfg["min_center_std"]
        if axis_ratio > max_axis or stdev > max_std:
            continue

        dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        pos_weight = 1.0 / (1.0 + dist_to_center / 100)
        shape_weight = 1.0 / (1.0 + abs(axis_ratio - 1.0))
        score = (shape_weight * cfg["axis_ratio_weight"] + pos_weight * cfg["center_bias_weight"]) * np.sqrt(area)

        # 新增时序运动惩罚：距离上一帧预测位置过远扣分
        if pred_prev_center is not None and pred_prev_speed > 1e-3:
            dist_pred = np.sqrt((cx - pred_prev_center[0])**2 + (cy - pred_prev_center[1])**2)
            max_allow_dist = max(35.0, pred_prev_speed * 3.5)
            if dist_pred > max_allow_dist:
                continue
            score -= dist_pred * 0.01

        if score > best_score:
            best_score = score
            best_ball = (cnt, (cx, cy), ellipse)
            best_rect = rect

    ball_center = None
    best_cnt = None
    if best_ball is not None:
        best_cnt, ball_center, ell = best_ball
        # 绘制椭圆、中心点、绿色识别包围框
        cv2.ellipse(frame, ell, (0, 255, 0), 2)
        cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 8, (0, 0, 255), -1)
        xr, yr, wr, hr = best_rect
        cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 255, 0), 2)

    return frame, ball_center, best_cnt

# ====================== PnP 三维解算 ======================
def pnp_3d_estimate(center_2d, cnt, cfg):
    if center_2d is None or cnt is None:
        return None
    obj_pts = np.array([
        [-cfg["ball_radius"], 0, 0],
        [cfg["ball_radius"], 0, 0],
        [0, -cfg["ball_radius"], 0],
        [0, cfg["ball_radius"], 0],
        [0, 0, -cfg["ball_radius"]],
        [0, 0, cfg["ball_radius"]]
    ], dtype=np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    img_pts = np.array([
        [cx - r, cy], [cx + r, cy],
        [cx, cy - r], [cx, cy + r],
        [cx, cy], [cx, cy]
    ], dtype=np.float32)
    ret, _, tvec = cv2.solvePnP(obj_pts, img_pts, cfg["camera_matrix"], cfg["dist_coeffs"])
    if ret:
        return tvec.flatten()
    return None

# ====================== 绘制运动信息 ======================
def draw_motion(frame, curr_pt, prev_pt, kf_pt, speed_scale, is_predicted=False):
    if curr_pt is None:
        return frame
    kf_x, kf_y = int(kf_pt[0]), int(kf_pt[1])
    cv2.circle(frame, (kf_x, kf_y), 6, (255, 0, 0), 2)

    label_text = "green ball predicted" if is_predicted else "green ball"
    cv2.putText(frame, label_text, (kf_x + 10, kf_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if prev_pt is not None:
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]
        speed = math.hypot(dx, dy)
        arrow_len = np.clip(speed * speed_scale * 3, 25, 140)
        norm = speed if speed > 1e-6 else 1.0
        end_x = int(curr_pt[0] + dx / norm * arrow_len)
        end_y = int(curr_pt[1] + dy / norm * arrow_len)
        cv2.arrowedLine(frame, (int(curr_pt[0]), int(curr_pt[1])), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(frame, f"Speed:{speed:.1f} px/frame",
                    (int(curr_pt[0])+10, int(curr_pt[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return frame

# ====================== 主函数 ======================
def main():
    cfg = load_config()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 创建视频输出目录
    video_dir = os.path.dirname(cfg["output_video"])
    os.makedirs(video_dir, exist_ok=True)

    # 2. 创建数据输出目录 & CSV 文件
    data_dir = os.path.dirname(cfg["output_data"])
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "speed.csv")

    cap = cv2.VideoCapture(cfg["input_video"])
    if not cap.isOpened():
        print("视频打开失败，请检查路径配置")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg["output_video"], fourcc, fps, (w, h))

    dt = 1.0 / fps
    kf = KalmanFilter2D(dt, cfg["kf_process_noise"], cfg["kf_measure_noise"])
    prev_center = None
    prev_speed = 0.0
    track_points = []
    lost_frames = 0
    frame_idx = 0
    use_predict = False
    predict_center = None

    # 初始化 CSV 并写入表头
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_s", "speed_pixel", "direction_deg"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_draw = frame.copy()
        current_time = round(frame_idx / fps, 3)

        # 传入上一帧预测坐标、速度用于时序择优惩罚
        frame_draw, curr_center, cnt = auto_filter_green_objects(frame_draw, cfg, pred_prev_center=prev_center, pred_prev_speed=prev_speed)

        # 遮挡短时匀速预测逻辑
        if curr_center is not None:
            lost_frames = 0
            use_predict = False
            predict_center = None
            pred_pt = kf.predict()
            kf_pt = kf.update(curr_center)
            output_center = curr_center
        else:
            lost_frames += 1
            pred_pt = kf.predict()
            # 未超限：匀速预测输出坐标
            if lost_frames <= cfg["max_lost_frame"] and prev_center is not None and prev_speed > 1e-3:
                use_predict = True
                predict_center = pred_pt
                output_center = predict_center
                cv2.putText(frame_draw, "green ball predicted", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)
            else:
                # 丢失超限，重置跟踪
                use_predict = False
                predict_center = None
                output_center = None
                cv2.putText(frame_draw, "green ball lost", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                kf = KalmanFilter2D(dt, cfg["kf_process_noise"], cfg["kf_measure_noise"])
                track_points.clear()
                lost_frames = 0

        # 计算速度、角度
        speed, angle = calc_speed_and_dir(prev_center, output_center)
        prev_speed = speed

        # PnP 解算
        tvec = pnp_3d_estimate(curr_center, cnt, cfg)
        if tvec is not None:
            cv2.putText(frame_draw, f"3D: {tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # 轨迹缓存绘制
        if output_center is not None:
            track_points.append((int(output_center[0]), int(output_center[1])))
            if len(track_points) > cfg["track_line_length"]:
                track_points.pop(0)
        if cfg["show_track_line"] and len(track_points) > 1:
            for i in range(1, len(track_points)):
                cv2.line(frame_draw, track_points[i-1], track_points[i], (255,0,255), 2)

        # 绘制箭头、文字
        frame_draw = draw_motion(frame_draw, output_center, prev_center, pred_pt, cfg["speed_scale"], is_predicted=use_predict)

        # CSV写入
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([frame_idx, current_time, speed, angle])

        # 更新状态
        prev_center = output_center
        out.write(frame_draw)
        cv2.imshow("Ball Track", frame_draw)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成！")
    print(f"结果视频: {cfg['output_video']}")
    print(f"帧速度数据: {csv_path}")

if __name__ == "__main__":
    main()
