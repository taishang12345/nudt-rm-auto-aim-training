import cv2
import numpy as np
import json
from pathlib import Path

def load_config():
    script_dir = Path(__file__).parent.resolve()
    homework_dir = script_dir.parent.resolve()
    repo_root = homework_dir.parent.parent.resolve()
    
    config_path = homework_dir / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config['video']['input_path'] = str(repo_root / config['video']['input_path'])
    return config

def analyze_components(frame, config):
    """
    分析所有连通域，可视化每个区域的 fill_ratio 和面积
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array(config['hsv']['lower_green'])
    upper = np.array(config['hsv']['upper_green'])
    mask = cv2.inRange(hsv, lower, upper)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config['morphology']['kernel_size'],) * 2
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                           iterations=config['morphology']['open_iterations'])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                           iterations=config['morphology']['close_iterations'])
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    h, w = frame.shape[:2]
    result = frame.copy()
    
    # 遍历所有连通域（跳过背景 0）
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # 获取该连通域的轮廓
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        cnt = contours[0]
        center, radius = cv2.minEnclosingCircle(cnt)
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        
        # 计算 fill_ratio
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        
        # 长宽比
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        
        # 判断是否通过筛选
        is_edge = (center[0] < radius or center[0] > w - radius or
                   center[1] < radius or center[1] > h - radius)
        
        min_fill = config['filter']['min_circularity']
        min_aspect = config['filter']['min_aspect_ratio']
        if is_edge:
            min_fill *= config['filter']['edge_relax_factor']
            min_aspect *= config['filter']['edge_relax_factor']
        
        passed = (fill_ratio > min_fill and aspect_ratio > min_aspect and
                  config['filter']['min_radius'] <= radius <= config['filter']['max_radius'] and
                  config['filter']['min_area'] <= area <= config['filter']['max_area'])
        
        # 绘制
        color = (0, 255, 0) if passed else (0, 0, 255)  # 绿色=通过，红色=未通过
        
        # 画外接矩形
        cv2.rectangle(result, (x, y), (x + width, y + height), color, 2)
        
        # 画最小外接圆
        cv2.circle(result, center, radius, (255, 0, 0), 2)
        
        # 画中心点
        cv2.circle(result, (cx, cy), 5, (255, 255, 0), -1)
        
        # 显示信息
        info = f"ID:{i} A:{area:.0f} R:{radius:.0f} F:{fill_ratio:.2f} Ar:{aspect_ratio:.2f}"
        cv2.putText(result, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 在终端打印
        status = "PASS" if passed else "FAIL"
        print(f"  ID={i}: area={area:.0f}, radius={radius:.0f}, fill_ratio={fill_ratio:.3f}, "
              f"aspect={aspect_ratio:.3f}, edge={is_edge} -> {status}")
    
    return result, mask

def main():
    config = load_config()
    
    cap = cv2.VideoCapture(config['video']['input_path'])
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("无法读取视频")
        return
    
    print(f"画面尺寸: {frame.shape[1]}x{frame.shape[0]}")
    print(f"HSV 阈值: {config['hsv']['lower_green']} ~ {config['hsv']['upper_green']}")
    print(f"筛选条件: fill_ratio > {config['filter']['min_circularity']}, "
          f"aspect > {config['filter']['min_aspect_ratio']}, "
          f"radius {config['filter']['min_radius']}~{config['filter']['max_radius']}, "
          f"area {config['filter']['min_area']}~{config['filter']['max_area']}")
    print("\n连通域分析:")
    
    result, mask = analyze_components(frame, config)
    
    # 保存结果
    script_dir = Path(__file__).parent.resolve()
    homework_dir = script_dir.parent.resolve()
    
    result_path = str(homework_dir / "data" / "connected_analysis.jpg")
    mask_path = str(homework_dir / "data" / "connected_mask.jpg")
    
    cv2.imwrite(result_path, result)
    cv2.imwrite(mask_path, mask)
    
    print(f"\n分析图已保存: {result_path}")
    print(f"Mask 图已保存: {mask_path}")
    
    # 显示
    cv2.imshow("Connected Components", result)
    cv2.imshow("Mask", mask)
    print("按 ESC 退出")
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()