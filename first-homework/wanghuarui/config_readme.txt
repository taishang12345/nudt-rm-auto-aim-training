配置文件说明

[video]
- input_path: 输入视频路径（相对于仓库根目录）
- output_path: 输出视频路径

[hsv]
- lower_green/upper_green: HSV绿色阈值范围

[morphology]
- kernel_size: 形态学核大小
- open_iterations: 开运算次数
- close_iterations: 闭运算次数

[filter]
- min_area/max_area: 面积过滤范围
- min_circularity: 最小圆度（0-1）
- min_aspect_ratio: 最小长宽比
- edge_relax_factor: 边缘放宽系数

[tracking]
- max_distance: 最大匹配距离（像素）
- max_radius_diff: 最大半径差
- max_lost_frames: 最大丢失帧数

[visualization]
- circle_color: 圆形框颜色 [B,G,R]
- arrow_color: 箭头颜色
- trail_color: 轨迹线颜色
- trail_length: 轨迹长度（帧数）
- arrow_scale: 箭头长度缩放系数

[csv_output]
- path: CSV输出路径