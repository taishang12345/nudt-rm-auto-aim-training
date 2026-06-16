import csv
import matplotlib.pyplot as plt

def read_csv(path):
    data = []
    with open(path,"r") as f:
        r = csv.DictReader(f)
        for row in r:
            data.append({
                "frame_idx":int(row["frame_idx"]),
                "speed":float(row["speed_pixel"]),
                "angle":float(row["direction_deg"])
            })
    return data

def min_angle_diff(a,b):
    diff = abs(a-b)
    return min(diff, 360-diff)

# 读取python与cpp两份csv
data_py = read_csv("output_data/speed.csv")
data_cpp = read_csv("cpp_output/frame_speed.csv")

speed_diff = []
angle_diff = []
frames = []
for py, cp in zip(data_py, data_cpp):
    frames.append(py["frame_idx"])
    speed_diff.append(py["speed"] - cp["speed"])
    angle_diff.append(min_angle_diff(py["angle"], cp["angle"])) # 环形角度修正

# 绘图
plt.figure(figsize=(12,9))
plt.subplot(2,1,1)
plt.scatter(frames, speed_diff, s=6, c="blue", label="speed_diff")
plt.title("speed_diff")
plt.grid()
plt.subplot(2,1,2)
plt.scatter(frames, angle_diff, s=6, c="blue", label="direction_diff")
plt.title("direction_diff (fixed circular angle)")
plt.grid()
plt.tight_layout()
plt.show()
