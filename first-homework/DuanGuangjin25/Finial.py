# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:57:56 2026

@author: jin
"""

import cv2
import numpy as np
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([77, 255, 255])
MIN_BALL_AREA = 100
ARROW_SCALE = 2
last_center = None
cap = cv2.VideoCapture(r'D:\Git\Git_work_space\first_homework.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break  
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_center = None
    max_area=0
    if cnts:
        for cnt in cnts:
            area=cv2.contourArea(cnt)
            if(area<100):
                continue
            (x,y),(a,b),angle=cv2.fitEllipse(cnt)
            ratio=max(a,b)/min(a,b)
            if(ratio<1.2 and area>max_area):
                best_cnt=cnt
                max_area=area
        if cv2.contourArea(best_cnt) > MIN_BALL_AREA:
            if best_cnt is not None:
                ((x, y), radius) = cv2.minEnclosingCircle(best_cnt)
                current_center = (int(x), int(y))
                cv2.circle(frame, current_center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, current_center, 5, (0, 0, 255), -1)
    if last_center is not None and current_center is not None:
        dx = current_center[0] - last_center[0]
        dy = current_center[1] - last_center[1]
        speed = np.sqrt(dx**2 + dy**2)
        if speed > 0.01:
            end_x = int(current_center[0] + dx * ARROW_SCALE)
            end_y = int(current_center[1] + dy * ARROW_SCALE)
            cv2.arrowedLine(frame, current_center, (end_x, end_y), (255, 0, 0), 2)
            cv2.putText(frame, f"Speed: {speed:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    last_center = current_center
    cv2.imshow("Green Ball Detection + Motion Arrow", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()