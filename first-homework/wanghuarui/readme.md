

# Week1 OpenCV Ball Recognition

**作者：** 王华睿 (GitHub： chaokaxb)  
**环境：** Python 3.10， OpenCV 4.x， NumPy  
**日期：** 2026-06-19

---

## 目录结构

```
first-homework/wanghuarui/
├── config.json              # 配置文件：HSV阈值、筛选参数、跟踪参数
├── config_readme.txt        # 配置说明文档：解释各参数含义
├── README.md                # 本文档
├── src/
│   ├── main.py              # 主程序：视频读取、绿球检测、跟踪、绘制、输出
│   ├── test.py              # 调试工具：10帧采样，框选分析HSV可行域
│   └── test_connected.py    # 调试工具：可视化所有连通域及筛选结果
├── video/                   # 输出视频目录(可删除以节省空间)
│   └── first-homework-annotated.mp4
└── data/                    # 输出数据目录
    └── tracking_data.csv    # 每帧的时间、速度、角度数据
```

> **注：** `video/` 目录下的输出视频为临时文件，提交前删除以节省仓库空间。

---

## 核心功能

- 检测视频中的绿色小球
- 使用圆形框标出小球位置
- 使用箭头表示小球运动方向
- 使用箭头长度和速度文本反映小球速度变化
- 支持短时遮挡预测(显示 ＂green ball predicted＂)
- 输出 CSV 数据用于定量分析

---

## 实现思路

### 1. HSV 色彩空间 + 动态阈值

传统固定阈值容易受光照影响。本实现采用**多帧采样 + 20/80 百分位数**确定 HSV 可行域：

- 从视频中均匀采样 10 帧
- 每帧框选绿色小球区域
- 统计框内绿色像素的 H、S、V 分布
- 用 **20/80 百分位数**替代 min/max，排除极端异常值
- 最终确定 HSV 阈值，兼顾鲁棒性和精确性

### 2. 连通域分析 + fill_ratio 圆度判定

传统圆度(4π·面积/周长²)受轮廓锯齿影响大。本实现采用：

- **连通域分析**(`connectedComponentsWithStats`)分离独立绿色区域
- **fill_ratio ＝ 面积 ／ 最小外接圆面积** 判定圆度
- 配合**长宽比**和**半径范围**筛选，有效区分小球与圆柱

### 3. 时序跟踪 + 遮挡预测

- 跨帧匹配时增加**距离约束**和**半径约束**，防止目标跳变
- 短时遮挡时启用**匀速外推预测**，保持跟踪连续性
- 长时间丢失后重置，等待重新识别

### 4. 边缘自适应

小球贴近画面边缘时，轮廓被裁切导致几何特征变形。实现中：
- 检测边缘状态
- 自动放宽筛选阈值
- 避免贴边时漏检

---

## 运行方式

### 前置要求

```bash
# 环境
Python 3.10+
OpenCV 4.x
NumPy

# 已配置的 Conda 环境
conda activate yolo
```

### 运行主程序

```bash
python first-homework/wanghuarui/src/main.py
```

**默认行为：**
- 输入视频： `first-homework/first-homework.mp4`
- 输出视频： `first-homework/wanghuarui/video/first-homework-annotated.mp4`
- 输出 CSV： `first-homework/wanghuarui/data/tracking_data.csv`

### 重新标定 HSV(可选)

如需针对新视频重新确定 HSV 阈值：

```bash
python first-homework/wanghuarui/src/test.py
```

按提示框选 10 帧中的绿色小球，程序自动输出推荐配置。

---

## 配置说明

详见 `config_readme.txt`。关键参数：

| 参数 | 说明 |
|---|---|
| `hsv.lower_green` ／ `upper_green` | HSV 绿色阈值，由 test.py 采样确定 |
| `filter.min_area` ／ `max_area` | 面积筛选范围，排除过小噪声和过大干扰 |
| `filter.min_circularity` | fill_ratio 阈值，区分圆形与细长形状 |
| `filter.min_radius` ／ `max_radius` | 半径范围，进一步约束目标大小 |
| `tracking.max_distance` | 跨帧最大匹配距离，防止跳变 |
| `tracking.max_lost_frames` | 最大丢失帧数，超过则重置跟踪 |

---

## 版本信息

- **Python：** 3.10.20 (conda env： yolo)
- **OpenCV：** 4.x
- **NumPy：** 1.x
- **PyTorch：** 2.5.1+cu124 (环境已配置，本作业未使用)

---

## 备注

- 输出视频文件较大，提交前建议删除 `video/` 目录内容
- CSV 数据可用于后续速度曲线分析和误差评估
- 本实现完全基于传统 CV 方法，未使用深度学习(YOLO 环境为后续作业准备)

---

## 开发迭代日志(DevLog)

本章节记录本次作业的开发迭代过程，供后续参考。

### 迭代 1：环境搭建与 Git 配置
- 确认 GitHub 用户名 `chaokaxb`，fork 来源 `wanghuarui`
- 配置 upstream 为 `Winter-Raymond/nudt-rm-auto-aim-training`
- 创建分支 `week1-opencv-ball`
- 搭建 Conda 环境 `yolo` (Python 3.10 + PyTorch 2.5.1 + CUDA 12.4)
- 安装 VSCode Python 扩展

### 迭代 2：基础框架与首次运行
- 创建目录结构 `first-homework/wanghuarui/`
- 编写 `config.json`、`main.py`、`README.md`
- 首次运行报错：`json.decoder.JSONDecodeError`(config.json 未保存)
- 修复后运行，检测到错误目标(白色物体)

### 迭代 3：HSV 阈值调试
- 问题：HSV 范围太宽，包含非绿色区域
- 尝试收紧 HSV 阈值，结果完全检测不到
- 意识到需要科学方法确定阈值

### 迭代 4：框选分析工具(test.py)
- 开发 `test.py`：从视频帧中框选绿色小球，分析 HSV 分布
- 发现圆度计算受轮廓锯齿影响(圆度仅 0.379)
- 提出 **fill_ratio ＝ 面积/外接圆面积** 替代传统圆度

### 迭代 5：多帧采样 + 百分位数
- 改进 `test.py`：均匀采样 10 帧，综合确定 HSV 可行域
- 引入 **20/80 百分位数**替代 min/max，排除压缩导致的色彩偏移
- 输出推荐 HSV 配置

### 迭代 6：连通域分析 + fill_ratio
- 修改 `main.py`：用 `connectedComponentsWithStats` 分离独立绿色区域
- 用 fill_ratio 判定圆度，配合长宽比筛选
- 添加半径约束(min_radius=50， max_radius=100)
- 有效区分小球与圆柱

### 迭代 7：调试与可视化
- 开发 `test_connected.py`：可视化所有连通域及筛选结果
- 发现 HSV 阈值仍有问题，白色圆柱被误检
- 收紧 S 下限(>116)，排除低饱和度白色

### 迭代 8：最终调优
- 根据 10 帧采样结果，确定最终 HSV 配置：
  - lower： [62， 116， 69]
  - upper： [88， 255， 208]
- 调整面积范围(8000-25000)匹配实际小球大小
- 测试通过，小球跟踪稳定，无跳变

### 迭代 9：文档整理与提交
- 编写完整 README.md
- 添加开发迭代日志
- 清理 video/ 目录临时文件
- 准备 Git 提交

---

## 关键设计决策

| 决策 | 方案 | 原因 |
|---|---|---|
| HSV 阈值确定 | 多帧采样 + 20/80 百分位 | 排除视频压缩导致的色彩偏移 |
| 圆度判定 | fill_ratio 替代 4πA/P² | 轮廓锯齿不影响判定 |
| 目标筛选 | 连通域分析 + 多条件联合 | 面积、半径、fill_ratio、长宽比共同约束 |
| 跟踪策略 | 距离约束 + 半径约束 + 预测 | 防止小球与圆柱间跳变 |
| 边缘处理 | 自适应放宽阈值 | 贴边时轮廓裁切不导致漏检 |

---

## 待改进方向

- [ ] HSV 阈值自动自适应(根据场景动态调整)
- [ ] S-V 相关性建模(亮暗变化时的函数关系)
- [ ] 卡尔曼滤波引入(当前为匀速预测，可优化)
- [ ] 多目标跟踪扩展(当前假设单目标)

