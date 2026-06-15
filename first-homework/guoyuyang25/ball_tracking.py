#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque

class BallTracker:
    def __init__(self):
        # 绿色的HSV范围
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([80, 255, 255])
        
        # 平滑用的历史位置
        self.history = deque(maxlen=5)
        self.prev_center = None
        self.trajectory = deque(maxlen=30)
        
        # 锁定跟踪
        self.locked = False
        self.locked_center = None
        self.search_radius = 150  # 搜索半径
    
    def circularity(self, contour):
        area = cv2.contourArea(contour)
        if area < 50:
            return 0
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return 4 * np.pi * area / (perimeter * perimeter)
    
    def detect_green_ball(self, frame, frame_count):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80:
                continue
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            candidates.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'area': area,
                'circularity': self.circularity(c)
            })
        
        if not candidates:
            return None, None
        
        # ========== 第一帧：选圆形度最高的（确保锁定球，不是圆柱体） ==========
        if not self.locked:
            # 按圆形度排序，选最圆的
            candidates.sort(key=lambda x: x['circularity'], reverse=True)
            best = candidates[0]
            self.locked = True
            self.locked_center = best['center']
            print(f"🔒 锁定球: {self.locked_center}, 圆形度: {best['circularity']:.3f}")
            return best['center'], best['radius']
        
        # ========== 锁定后：只在锁定位置附近搜索，不要求圆形度 ==========
        # 按距离排序
        for c in candidates:
            dx = c['center'][0] - self.locked_center[0]
            dy = c['center'][1] - self.locked_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < self.search_radius:
                # 找到了，更新锁定位置
                self.locked_center = c['center']
                return c['center'], c['radius']
        
        # 没找到，返回None（不预测，避免画错）
        return None, None
    
    def smooth_position(self, center):
        if center is None:
            return None
        self.history.append(center)
        if len(self.history) < 2:
            return center
        avg_x = int(np.mean([p[0] for p in self.history]))
        avg_y = int(np.mean([p[1] for p in self.history]))
        return (avg_x, avg_y)
    
    def draw_arrow(self, frame, from_center, to_center, speed):
        if from_center is None or to_center is None:
            return
        x1, y1 = from_center
        x2, y2 = to_center
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return
        arrow_length = min(max(speed * 1.5, 10), 80)
        magnitude = np.sqrt(dx**2 + dy**2)
        dir_x = dx / magnitude * arrow_length
        dir_y = dy / magnitude * arrow_length
        end_x = int(x2 + dir_x)
        end_y = int(y2 + dir_y)
        if speed < 5:
            color = (255, 0, 0)
        elif speed < 15:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        thickness = min(int(speed / 5) + 1, 5)
        cv2.arrowedLine(frame, (x2, y2), (end_x, end_y), color, thickness, tipLength=0.3)
        cv2.putText(frame, f"v: {speed:.1f}", (x2 + 10, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, input_path, output_path=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {input_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息：{width}x{height}, {fps:.2f}fps, {total_frames}帧")
        
        if output_path is None:
            output_path = "output_with_arrows.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        prev_smooth = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            center, radius = self.detect_green_ball(frame, frame_count)
            
            if center is not None:
                smooth_center = self.smooth_position(center)
                
                cv2.circle(frame, smooth_center, radius, (0, 255, 0), 2)
                cv2.circle(frame, smooth_center, 3, (0, 0, 255), -1)
                
                self.trajectory.append(smooth_center)
                if len(self.trajectory) >= 2:
                    points = list(self.trajectory)
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)
                
                if prev_smooth is not None:
                    dx = smooth_center[0] - prev_smooth[0]
                    dy = smooth_center[1] - prev_smooth[1]
                    speed = np.sqrt(dx**2 + dy**2)
                    if speed > 0.5:
                        self.draw_arrow(frame, prev_smooth, smooth_center, speed)
                
                prev_smooth = smooth_center
                
                # 显示搜索范围
                cv2.circle(frame, self.locked_center, self.search_radius, (0, 255, 0), 1)
                cv2.putText(frame, "TRACKING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 没找到，不画任何东西，也不显示lost
                cv2.putText(frame, "SEARCHING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, self.locked_center, self.search_radius, (0, 255, 255), 1)
            
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            cv2.imshow('Ball Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ 完成: {output_path}")


if __name__ == "__main__":
    input_video = "first-homework/first-homework.mp4"
    output_video = "first-homework/output_with_arrows.mp4"
    
    tracker = BallTracker()
    tracker.process_video(input_video, output_video)
