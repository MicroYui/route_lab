# -*- coding: utf-8 -*-
# plotter.py
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 移除3D绘图相关的导入
import numpy as np
import os
from config import AREA_WIDTH, AREA_HEIGHT, DATA_CENTER_POSITION, UAV_MIN_ENERGY  # DATA_CENTER_POSITION is 2D

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败: {e}. 部分中文可能无法正确显示。")


def plot_uav_trajectories_2d(simulation_log, num_uavs, output_dir="plots"):
    if not simulation_log: print("没有仿真日志数据可供绘制轨迹。"); return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    plt.figure(figsize=(10, 10))
    trajectories_x = {i: [] for i in range(num_uavs)}
    trajectories_y = {i: [] for i in range(num_uavs)}

    for log_entry in simulation_log:
        positions = log_entry.get("uav_positions", {})  # positions 现在是 {id: [x,y]}
        for uav_id, pos_xy in positions.items():
            if uav_id in trajectories_x and len(pos_xy) == 2:  # 确保是二维数据
                trajectories_x[uav_id].append(pos_xy[0])
                trajectories_y[uav_id].append(pos_xy[1])

    colors = plt.cm.get_cmap('gist_rainbow', num_uavs) if num_uavs > 0 else []

    for i in range(num_uavs):
        if trajectories_x[i] and trajectories_y[i]:
            color_val = colors(i / num_uavs) if num_uavs > 0 else 'blue'
            plt.plot(trajectories_x[i], trajectories_y[i], marker='o', markersize=1, linestyle='-', color=color_val,
                     label=f'UAV {i}' if num_uavs <= 10 else None, alpha=0.6)
            plt.scatter(trajectories_x[i][0], trajectories_y[i][0], marker='s', color=color_val, s=30, alpha=0.8)
            plt.scatter(trajectories_x[i][-1], trajectories_y[i][-1], marker='x', color=color_val, s=50, alpha=0.8)

    dc_pos_xy = DATA_CENTER_POSITION[:2]  # 获取二维数据中心位置
    plt.scatter(dc_pos_xy[0], dc_pos_xy[1], marker='H', color='red', s=150, label='数据中心 (D)', zorder=5)

    plt.title('所有无人机的2D运动轨迹')
    plt.xlabel('X 坐标 (米)')
    plt.ylabel('Y 坐标 (米)')
    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, AREA_HEIGHT)
    if num_uavs <= 10: plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.85 if num_uavs <= 10 else 1, 1])

    filename = os.path.join(output_dir, "uav_trajectories_2d.png")
    plt.savefig(filename)
    print(f"2D轨迹图已保存到: {filename}")


# plot_uav_trajectories_3d 函数可以被完全删除

# plot_metric_over_time 函数逻辑不变，因为它不直接处理位置维度
def plot_metric_over_time(simulation_log, num_uavs, metric_key, title, ylabel, output_dir="plots", y_min_line=None,
                          y_min_label=""):
    if not simulation_log: print(f"没有仿真日志数据可供绘制 {title}。"); return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    plt.figure(figsize=(12, 7))
    times = [log_entry["时间 (s)"] for log_entry in simulation_log]
    metric_data = {i: [] for i in range(num_uavs)}

    for log_entry in simulation_log:
        data_dict = log_entry.get(metric_key, {})
        for uav_id in range(num_uavs):  # 确保为每个UAV尝试获取数据
            value = data_dict.get(uav_id)  # 使用 .get 防止 KeyError
            if value is not None:
                metric_data[uav_id].append(value)
            elif metric_data[uav_id]:  # 如果之前有数据，现在没有了，可以填充NaN或前一个值
                metric_data[uav_id].append(metric_data[uav_id][-1])  # 简单填充前一个值
            # else: 如果一开始就没有数据，列表将为空

    colors = plt.cm.get_cmap('gist_rainbow', num_uavs) if num_uavs > 0 else []

    for i in range(num_uavs):
        current_uav_data = metric_data[i]
        # 确保数据长度与时间序列匹配，或者只绘制有效部分
        if current_uav_data:
            # 如果数据长度少于times长度，说明UAV可能提前“死亡”或停止记录
            effective_times = times[:len(current_uav_data)]
            plt.plot(effective_times, current_uav_data, linestyle='-',
                     color=colors(i / num_uavs) if num_uavs > 0 else 'blue',
                     label=f'UAV {i}' if num_uavs <= 10 else None, alpha=0.8)

    if y_min_line is not None:
        plt.axhline(y=y_min_line, color='r', linestyle='--',
                    label=y_min_label if y_min_label else f'阈值: {y_min_line}')

    plt.title(title);
    plt.xlabel('时间 (s)');
    plt.ylabel(ylabel)
    if num_uavs <= 10:
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    else:
        print(f"为 {title} 图表省略了UAV图例，因为UAV数量 ({num_uavs}) 过多。")
    plt.grid(True);
    plt.tight_layout(rect=[0, 0, 0.85 if num_uavs <= 10 else 1, 1])

    filename = os.path.join(output_dir, f"{metric_key.replace(' ', '_')}_over_time.png")
    plt.savefig(filename);
    print(f"{title} 图已保存到: {filename}")


def show_all_plots(): plt.show()