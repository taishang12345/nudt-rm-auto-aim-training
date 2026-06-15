# Week1 OpenCV Ball Recognition 
**亓涵玉**

## 1. 概述

程序完成以下功能：

1. 检测视频中的绿色小球；
2. 使用圆形框标出小球位置；
3. 使用箭头表示小球运动方向；
4. 使用箭头长度和速度文本反映小球速度变化。
---

## 2. 目录结构

```text
first-homework/Qi-huanye/
├── CMakeLists.txt
├── README.md
├── src/
│   └── main.cpp
└── first-homework-annotated.mp4
```

各文件说明如下：

- `src/main.cpp`：作业主程序，负责视频读取、绿球检测、轨迹匹配和结果绘制；
- `CMakeLists.txt`：独立编译配置文件；
- `README.md`：作业说明文档；

---

## 3. 开发环境

- Ubuntu Linux
- CMake >= 3.10
- C++17
- OpenCV 4

---

## 4. 构建方式

在仓库根目录执行：

```bash
cmake -S first-homework/Qi-huanye -B first-homework/Qi-huanye/build
cmake --build first-homework/Qi-huanye/build -j
```

编译完成后会生成可执行文件：

- `first-homework/Qi-huanye/build/ball_tracker`

---

## 5. 运行方式

### 5.1 使用默认输入输出路径

在仓库根目录执行：

```bash
./first-homework/Qi-huanye/build/ball_tracker
```

默认行为如下：

- 输入视频：`first-homework/first-homework.mp4`
- 输出视频：`first-homework/Qi-huanye/first-homework-annotated.mp4`

### 5.2 指定输入输出路径

```bash
./first-homework/Qi-huanye/build/ball_tracker \
  first-homework/first-homework.mp4 \
  first-homework/Qi-huanye/first-homework-annotated.mp4
```

---

## 6. 实现方案

### 6.1 绿色区域提取

程序首先将输入图像从 BGR 空间转换到 HSV 空间，并通过绿色阈值提取候选区域。

同时，程序对二值图进行了开运算和闭运算，用于降低噪声：

### 6.2 候选目标筛选

视频中同时存在绿色圆柱和绿色小球，仅靠颜色无法完成区分，因此程序进一步使用几何特征进行筛选

### 6.3 跨帧匹配与跟踪

在每一帧中，程序会对候选目标计算形状分数，并结合上一帧状态进行匹配

### 6.4 速度与方向绘制

程序通过相邻两帧中心点差值估计速度向量，并据此绘制箭头

---

## 7. 关键问题与处理

### 7.1 绿色圆柱干扰

问题：

- 视频中同时存在绿色圆柱；
- 如果只做颜色分割，会把圆柱也当成目标。

处理方法：

- 增加圆度、长宽比、面积等几何约束；
- 增加跨帧匹配限制，避免目标在小球和圆柱之间跳变。

### 7.2 小球贴近画面边缘时丢失

问题：

- 小球靠近上边缘时，轮廓会被裁切；
- 原始筛选条件过严时，可能把真实小球误判为无效目标。

处理方法：

- 对贴边目标放宽几何约束；
- 放宽面积上限，避免贴边阶段因轮廓变化导致漏检；
- 增加目标切换约束，防止误切换到圆柱。

### 7.3 速度显示不稳定

问题：

- 若目标匹配错误，速度会出现异常突变；
- 这会直接影响箭头方向和长度。

处理方法：

- 在跨帧匹配时增加距离约束和半径约束；
- 对明显不合理的候选目标直接拒绝。