"""
Week1 OpenCV Ball Recognition
检测视频中的绿色小球，绘制圆形框、运动箭头、速度文本
输出标注视频和 CSV 数据
"""

import cv2          # OpenCV: 计算机视觉库，处理图像和视频
import json         # json: 读取配置文件
import numpy as np  # numpy: 数值计算，处理数组和矩阵
import csv          # csv: 写入逗号分隔数据文件
import os           # os: 操作系统接口，创建目录等
from pathlib import Path      # Path: 跨平台路径处理
from collections import deque # deque: 双端队列，用于缓存轨迹点


def load_config():
    """
    加载配置文件，自动处理路径
    作用：让程序在任何电脑上都能正确找到视频和输出位置
    """
    # __file__ 是当前文件路径（main.py）
    # .parent 获取所在文件夹（src/）
    # .resolve() 转为绝对路径
    script_dir = Path(__file__).parent.resolve()
    
    # 向上退一级：从 src/ 到 wanghuarui/
    homework_dir = script_dir.parent.resolve()
    
    # 再向上退一级：从 wanghuarui/ 到 first-homework/
    first_homework_dir = homework_dir.parent.resolve()
    
    # 再向上退一级：从 first-homework/ 到仓库根目录
    repo_root = first_homework_dir.parent.resolve()
    
    # 打开 config.json 配置文件
    config_path = homework_dir / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 输入视频路径：从仓库根目录拼接
    # 例如：~/nudt-rm-auto-aim-training/first-homework/first-homework.mp4
    config['video']['input_path'] = str(repo_root / config['video']['input_path'])
    
    # 输出视频路径：固定到 wanghuarui/video/ 下
    config['video']['output_path'] = str(homework_dir / "video" / "first-homework-annotated.mp4")
    
    # CSV 输出路径：固定到 wanghuarui/data/ 下
    config['csv_output']['path'] = str(homework_dir / "data" / "tracking_data.csv")
    
    # 自动创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(config['video']['output_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['csv_output']['path']), exist_ok=True)
    
    return config


def detect_green_ball(frame, config):
    """
    检测绿色小球
    作用：从视频帧中找出绿色小球的候选位置
    步骤：HSV颜色过滤 -> 形态学去噪 -> 连通域分析 -> 几何筛选
    """
    # 将 BGR 图像转为 HSV 色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 根据 config 中的阈值，提取绿色区域
    lower = np.array(config['hsv']['lower_green'])
    upper = np.array(config['hsv']['upper_green'])
    mask = cv2.inRange(hsv, lower, upper)
    
    # 形态学操作：开运算去噪，闭运算填补空洞
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config['morphology']['kernel_size'],) * 2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel,
        iterations=config['morphology']['open_iterations']
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel,
        iterations=config['morphology']['close_iterations']
    )
    
    # 连通域分析：找到绿色连续区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    candidates = []
    h, w = frame.shape[:2]
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < config['filter']['min_area'] or area > config['filter']['max_area']:
            continue
        
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        cnt = contours[0]
        
        center, radius = cv2.minEnclosingCircle(cnt)
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        
        if radius < config['filter']['min_radius'] or radius > config['filter']['max_radius']:
            continue
        
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        
        is_edge = (
            center[0] < radius or center[0] > w - radius or
            center[1] < radius or center[1] > h - radius
        )
        
        min_fill = config['filter']['min_circularity']
        min_aspect = config['filter']['min_aspect_ratio']
        if is_edge:
            min_fill *= config['filter']['edge_relax_factor']
            min_aspect *= config['filter']['edge_relax_factor']
        
        if fill_ratio > min_fill and aspect_ratio > min_aspect:
            candidates.append({
                'center': center,
                'radius': radius,
                'area': area,
                'fill_ratio': fill_ratio
            })
    
    return candidates


def main():
    """
    主程序
    作用：读取视频 -> 逐帧检测 -> 跟踪 -> 绘制 -> 输出
    """
    # 加载配置
    config = load_config()
    
    # 打开输入视频
    cap = cv2.VideoCapture(config['video']['input_path'])
    if not cap.isOpened():
        print(f"错误：无法打开视频 {config['video']['input_path']}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)           # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 高度
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    
    # 创建视频写入器，输出标注后的视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    out = cv2.VideoWriter(
        config['video']['output_path'], 
        fourcc, fps, (width, height)
    )
    
    # 跟踪状态变量
    trail = deque(maxlen=config['visualization']['trail_length'])  # 轨迹缓存
    prev_center = None    # 上一帧小球中心
    lost_count = 0        # 连续丢失帧数
    max_lost = config['tracking']['max_lost_frames']  # 最大允许丢失帧数
    
    csv_data = []         # 存储每帧数据，最后写入CSV
    frame_idx = 0         # 帧计数器
    
    # 逐帧处理视频
    while True:
        ret, frame = cap.read()  # 读取一帧
        if not ret:
            break  # 视频结束
        
        # 检测绿色小球候选
        candidates = detect_green_ball(frame, config)
        
        # 从候选中选出最佳目标
        target = None
        if candidates:
            if prev_center:
                # 有上一帧位置时，选距离最近的
                candidates.sort(key=lambda c: np.linalg.norm(
                    np.array(c['center']) - np.array(prev_center)
                ))
                best = candidates[0]
                dist = np.linalg.norm(np.array(best['center']) - np.array(prev_center))
                
                # 距离太远则认为是干扰，放弃匹配
                if dist < config['tracking']['max_distance']:
                    target = best
            else:
                # 第一帧，选面积最大的
                target = max(candidates, key=lambda c: c['area'])
        
        # 更新跟踪状态
        predicted = False  # 是否使用预测位置
        
        if target:
            # 检测到目标，重置丢失计数
            lost_count = 0
            center = target['center']
            radius = target['radius']
            trail.append(center)
            prev_center = center
        
        elif lost_count < max_lost and prev_center and len(trail) >= 2:
            # 短时丢失，用匀速外推预测位置
            # 假设小球保持之前的运动方向和速度
            dx = trail[-1][0] - trail[-2][0]  # X方向位移
            dy = trail[-1][1] - trail[-2][1]  # Y方向位移
            center = (prev_center[0] + dx, prev_center[1] + dy)
            radius = 20  # 预测时不知道半径，用默认值
            trail.append(center)
            prev_center = center
            lost_count += 1
            predicted = True  # 标记为预测状态
        
        else:
            # 长时间丢失，重置跟踪
            prev_center = None
            trail.clear()
        
        # 计算速度
        velocity, angle = 0.0, 0.0
        if prev_center and len(trail) >= 2:
            dx = trail[-1][0] - trail[-2][0]
            dy = trail[-1][1] - trail[-2][1]
            dt = 1.0 / fps  # 两帧之间的时间间隔
            
            # 速度 = 位移 / 时间
            velocity = np.sqrt(dx**2 + dy**2) / dt
            
            # 角度：0度为右，90度为上，180度为左，270度为下
            # Y轴向下，所以dy取负
            angle = np.degrees(np.arctan2(-dy, dx))
            if angle < 0:
                angle += 360  # 转为0-360度
        
        # 绘制标注
        annotated = frame.copy()  # 复制原帧，避免修改
        
        if prev_center:
            # 绘制轨迹线（粉色）
            if len(trail) > 1:
                points = np.array(list(trail), np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated, [points], False,
                    tuple(config['visualization']['trail_color']), 2
                )
            
            # 绘制圆形框（真实检测时）
            if not predicted:
                cv2.circle(
                    annotated, prev_center, radius,
                    tuple(config['visualization']['circle_color']), 2
                )
            
            # 绘制中心红点
            cv2.circle(annotated, prev_center, 5, (0, 0, 255), -1)
            
            # 绘制速度箭头
            if velocity > 1:
                scale = config['visualization']['arrow_scale']
                # 计算箭头终点
                dx = int(np.cos(np.radians(angle)) * velocity * scale / fps)
                dy = -int(np.sin(np.radians(angle)) * velocity * scale / fps)
                end = (prev_center[0] + dx, prev_center[1] + dy)
                
                cv2.arrowedLine(
                    annotated, prev_center, end,
                    tuple(config['visualization']['arrow_color']), 2,
                    cv2.LINE_AA, 0, 0.3
                )
                
                # 绘制速度文本
                text = f"v: {velocity:.1f} px/s"
                cv2.putText(
                    annotated, text,
                    (prev_center[0], prev_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
            
            # 预测状态标记
            if predicted:
                cv2.putText(
                    annotated, "green ball predicted",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2
                )
        
        # 绘制帧计数
        cv2.putText(
            annotated, f"Frame: {frame_idx}/{total_frames}",
            (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (255, 255, 255), 1
        )
        
        # 写入输出视频
        out.write(annotated)
        
        # 记录 CSV 数据
        csv_data.append({
            'frame': frame_idx,
            'time': round(frame_idx / fps, 3),
            'velocity': round(velocity, 2),
            'angle': round(angle, 2),
            'predicted': predicted
        })
        
        frame_idx += 1
        
        # 调试用：显示处理窗口（可选）
        # cv2.imshow('Tracking', annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # 释放资源
    cap.release()
    out.release()
    # cv2.destroyAllWindows()  # 如果用了 imshow 需要释放
    
    # 写入 CSV 文件
    with open(config['csv_output']['path'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=['frame', 'time', 'velocity', 'angle', 'predicted']
        )
        writer.writeheader()
        writer.writerows(csv_data)
    
    # 输出结果
    print(f"完成！处理了 {frame_idx} 帧。")
    print(f"输出视频：{config['video']['output_path']}")
    print(f"CSV数据：{config['csv_output']['path']}")


# 程序入口
if __name__ == '__main__':
    main()