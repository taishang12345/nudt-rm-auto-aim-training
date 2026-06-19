import cv2
import numpy as np
from pathlib import Path

def load_config():
    script_dir = Path(__file__).parent.resolve()
    homework_dir = script_dir.parent.resolve()
    repo_root = homework_dir.parent.parent.resolve()
    
    video_path = str(repo_root / "first-homework" / "first-homework.mp4")
    return video_path, homework_dir

def analyze_frame(frame, frame_idx, total_frames):
    """
    分析单帧，框选绿色小球区域
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    
    print(f"\n{'='*50}")
    print(f"第 {frame_idx+1}/{total_frames} 帧 (时间: {frame_idx/total_frames*100:.1f}%)")
    print(f"画面尺寸: {w}x{h}")
    print("操作: 鼠标左键按下开始框选，拖动调整，释放完成")
    print("       框选绿色小球区域，按 ESC 跳过此帧")
    
    # 保存当前帧供参考
    output_path = str(homework_dir / "data" / f"frame_{frame_idx:03d}.jpg")
    cv2.imwrite(output_path, frame)
    
    # 框选状态
    drawing = False
    ix, iy = -1, -1
    ex, ey = -1, -1
    
    result = {
        'frame_idx': frame_idx,
        'hsv_stats': None,
        'area': None,
        'radius': None,
        'fill_ratio': None
    }
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, ex, ey
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            ex, ey = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            ex, ey = x, y
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            ex, ey = x, y
            
            x1, y1 = min(ix, ex), min(iy, ey)
            x2, y2 = max(ix, ex), max(iy, ey)
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                print("框选区域太小，忽略")
                return
            
            # 提取框内区域
            roi_hsv = hsv[y1:y2, x1:x2]
            
            # 宽阈值预筛选绿色
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(roi_hsv, lower_green, upper_green)
            
            green_pixels = cv2.countNonZero(mask)
            total_pixels = mask.size
            
            print(f"\n框选区域: ({x1}, {y1}) 到 ({x2}, {y2})")
            print(f"区域大小: {x2-x1}x{y2-y1}")
            print(f"绿色像素数: {green_pixels} / {total_pixels} ({green_pixels/total_pixels*100:.1f}%)")
            
            if green_pixels == 0:
                print("未检测到绿色像素")
                return
            
            # 获取绿色像素的 HSV 值
            green_hsv = roi_hsv[mask > 0]
            
            # 计算统计值
            h_min, h_max = int(np.percentile(green_hsv[:,0], 20)), int(np.percentile(green_hsv[:,0], 80))
            s_min, s_max = int(np.percentile(green_hsv[:,1], 20)), int(np.percentile(green_hsv[:,1], 80))
            v_min, v_max = int(np.percentile(green_hsv[:,2], 20)), int(np.percentile(green_hsv[:,2], 80))
            
            print(f"\n框内绿色像素 HSV 统计:")
            print(f"  H: min={h_min}, max={h_max}, mean={green_hsv[:,0].mean():.1f}")
            print(f"  S: min={s_min}, max={s_max}, mean={green_hsv[:,1].mean():.1f}")
            print(f"  V: min={v_min}, max={v_max}, mean={green_hsv[:,2].mean():.1f}")
            
            # 保存结果
            result['hsv_stats'] = {
                'h': (h_min, h_max),
                's': (s_min, s_max),
                'v': (v_min, v_max)
            }
            
            # 分析轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                center, radius = cv2.minEnclosingCircle(largest)
                
                result['area'] = area
                result['radius'] = radius
                result['fill_ratio'] = area / (np.pi * radius * radius) if radius > 0 else 0
                
                print(f"\n最大绿色轮廓:")
                print(f"  面积: {area:.1f}")
                print(f"  圆度: {circularity:.3f}")
                print(f"  外接圆半径: {radius:.1f}")
                print(f"  fill_ratio: {result['fill_ratio']:.3f}")
            
            # 保存分析图
            analysis = frame.copy()
            cv2.rectangle(analysis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            analysis_path = str(homework_dir / "data" / f"analysis_{frame_idx:03d}.jpg")
            cv2.imwrite(analysis_path, analysis)
            print(f"\n分析图已保存: {analysis_path}")
    
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)
    
    while True:
        display = frame.copy()
        if drawing or (ix != -1 and ex != -1):
            x1, y1 = min(ix, ex), min(iy, ey)
            x2, y2 = max(ix, ex), max(iy, ey)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(display, f"Frame {frame_idx+1}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "ESC=Skip, Drag=Select", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow('Frame', display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("跳过此帧")
            break
    
    cv2.destroyAllWindows()
    return result

def main():
    global homework_dir
    video_path, homework_dir = load_config()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频总帧数: {total_frames}")
    print("将均匀采样 10 帧进行分析")
    
    # 均匀采样 10 帧
    sample_indices = [int(i * total_frames / 10) for i in range(10)]
    print(f"采样帧索引: {sample_indices}")
    
    results = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"无法读取第 {idx} 帧")
            continue
        
        result = analyze_frame(frame, idx, total_frames)
        if result['hsv_stats'] is not None:
            results.append(result)
    
    cap.release()
    
    # 汇总所有帧的数据
    if not results:
        print("\n没有有效数据")
        return
    
    print(f"\n{'='*50}")
    print("汇总所有帧的 HSV 可行域 (20/80百分位):")
    
    h_mins = [r['hsv_stats']['h'][0] for r in results]
    h_maxs = [r['hsv_stats']['h'][1] for r in results]
    s_mins = [r['hsv_stats']['s'][0] for r in results]
    s_maxs = [r['hsv_stats']['s'][1] for r in results]
    v_mins = [r['hsv_stats']['v'][0] for r in results]
    v_maxs = [r['hsv_stats']['v'][1] for r in results]
    
    # 20/80 百分位数汇总
    h_lower = int(np.percentile(h_mins, 20))
    h_upper = int(np.percentile(h_maxs, 80))
    s_lower = int(np.percentile(s_mins, 20))
    s_upper = int(np.percentile(s_maxs, 80))
    v_lower = int(np.percentile(v_mins, 20))
    v_upper = int(np.percentile(v_maxs, 80))
    
    # 10/90 百分位数（更宽松）
    h_lower_loose = int(np.percentile(h_mins, 10))
    h_upper_loose = int(np.percentile(h_maxs, 90))
    s_lower_loose = int(np.percentile(s_mins, 10))
    s_upper_loose = int(np.percentile(s_maxs, 90))
    v_lower_loose = int(np.percentile(v_mins, 10))
    v_upper_loose = int(np.percentile(v_maxs, 90))
    
    print(f"\n20/80 百分位 (推荐):")
    print(f'  "lower_green": [{h_lower}, {s_lower}, {v_lower}],')
    print(f'  "upper_green": [{h_upper}, {s_upper}, {v_upper}]')
    
    print(f"\n10/90 百分位 (宽松):")
    print(f'  "lower_green": [{h_lower_loose}, {s_lower_loose}, {v_lower_loose}],')
    print(f'  "upper_green": [{h_upper_loose}, {s_upper_loose}, {v_upper_loose}]')
    
    # 推荐配置：20/80 + 边距，S下限确保>50
    print(f"\n推荐配置 (20/80 + 边距, S>50):")
    print(f'  "lower_green": [{max(0, h_lower-5)}, {max(50, s_lower-20)}, {max(0, v_lower-20)}],')
    print(f'  "upper_green": [{min(180, h_upper+5)}, {min(255, s_upper+20)}, {min(255, v_upper+20)}]')
    
    # 面积和半径统计
    areas = [r['area'] for r in results if r['area'] is not None]
    radii = [r['radius'] for r in results if r['radius'] is not None]
    fill_ratios = [r['fill_ratio'] for r in results if r['fill_ratio'] is not None]
    
    if areas:
        print(f"\n面积统计: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}")
        print(f"半径统计: min={min(radii):.1f}, max={max(radii):.1f}, mean={np.mean(radii):.1f}")
        print(f"fill_ratio统计: min={min(fill_ratios):.3f}, max={max(fill_ratios):.3f}, mean={np.mean(fill_ratios):.3f}")
    
    print(f"\n{'='*50}")
    print("分析完成！请使用推荐配置修改 config.json")
    
if __name__ == '__main__':
    main()