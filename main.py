# -*- coding: utf-8 -*-
# main.py
import random
import time
import numpy as np
import json
from config import (N_UAVS, TOTAL_SIMULATION_TIME, UAV_MAX_ENERGY, UAV_MIN_ENERGY,
                    DELTA_T, DEBUG_LEVEL, DECISION_ALGORITHM, AREA_HEIGHT, SIMULATION_SEED)
from simulation_environment import SimulationEnvironment

# 尝试导入绘图模块，如果失败则禁用绘图功能
try:
    import plotter
    plotter_exists = True
except ImportError:
    plotter_exists = False
    print("警告: 未找到 matplotlib 或 plotter.py 模块，图形化展示功能将被禁用。")
    print("请尝试安装 matplotlib: pip install matplotlib")


def main():
    if SIMULATION_SEED is not None:
        print(f"信息: 使用固定的随机数种子: {SIMULATION_SEED}")
        random.seed(SIMULATION_SEED)
        np.random.seed(SIMULATION_SEED)
    else:
        print("信息: 未使用固定的随机数种子，每次运行结果可能不同。")

    print("开始无人机网络路由与能耗仿真...")
    simulation_start_real_time = time.time()

    env = SimulationEnvironment(num_uavs=N_UAVS)

    simulation_is_active = True
    while simulation_is_active:
        simulation_is_active = env.run_step()

        if env.current_time >= TOTAL_SIMULATION_TIME:
            if DEBUG_LEVEL >= 0: print(f"达到预设的总仿真时间: {TOTAL_SIMULATION_TIME}s.")
            break

        if not any(uav.energy >= UAV_MIN_ENERGY for uav in env.uavs):
            if DEBUG_LEVEL >= 0: print(
                f"所有无人机电量均低于最低安全阈值或耗尽，仿真提前结束于 {env.current_time:.2f}s.")
            break

    simulation_end_real_time = time.time()
    total_real_execution_time = simulation_end_real_time - simulation_start_real_time

    print(f"\n仿真结束. 模拟时长: {env.current_time:.2f}秒. 实际运行耗时: {total_real_execution_time:.2f}秒.")

    if not env.simulation_log:
        print("没有仿真日志数据可供分析。")
        return

    final_log_entry = env.simulation_log[-1]
    print("\n最终时刻的性能指标汇总:")
    for key, value in final_log_entry.items():
        if key not in ["uav_positions", "uav_queues_kb", "uav_energies"]:
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    total_bits_to_dc = env.data_center.total_data_received_bits
    print(f"  -> 网络总吞吐量 (传到数据中心总量): {total_bits_to_dc / (8 * 1024 * 1024):.3f} MB")

    final_normalized_energies = [(uav.energy / (float(UAV_MAX_ENERGY) if UAV_MAX_ENERGY > 0 else 1.0)) for uav in
                                 env.uavs]
    if final_normalized_energies:
        final_energy_balance_std_dev = np.std(final_normalized_energies)
        print(f"  -> T时刻能量均衡度 (归一化能量标准差): {final_energy_balance_std_dev:.4f}")

    actual_network_lifetime = env.current_time
    for log_item in env.simulation_log:
        first_dead_time = log_item.get("首个UAV低电量时间 (s)", -1.0)
        if first_dead_time != -1.0:
            actual_network_lifetime = first_dead_time
            break
    print(f"  -> 网络实际生存时间 (首个UAV低电量): {actual_network_lifetime:.2f} 秒")

    output_plot_dir = "simulation_plots"

    try:
        log_filename = f"{DECISION_ALGORITHM}_log_N{N_UAVS}_T{int(TOTAL_SIMULATION_TIME)}.json"
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(env.simulation_log, f, ensure_ascii=False, indent=4)
        print(f"详细仿真日志已保存到: {log_filename}")
    except Exception as e:
        print(f"保存日志文件失败: {e}")

    # --- 调用绘图函数 ---
    # 将此代码块移出上面的 try-except 块的 except 分支
    if plotter_exists and env.simulation_log:
        print("\n正在生成图表...")
        try:
            plotter.plot_uav_trajectories_2d(env.simulation_log, N_UAVS, output_dir=output_plot_dir)
            # plotter.plot_uav_trajectories_3d(...) # 3D图已移除或注释

            plotter.plot_metric_over_time(env.simulation_log, N_UAVS,
                                          metric_key="uav_queues_kb",
                                          title="各UAV队列长度变化 (KB)",
                                          ylabel="队列长度 (KB)",
                                          output_dir=output_plot_dir)

            plotter.plot_metric_over_time(env.simulation_log, N_UAVS,
                                          metric_key="uav_energies",
                                          title="各UAV能量变化",
                                          ylabel=f"剩余能量 (单位 E_max={UAV_MAX_ENERGY})",
                                          output_dir=output_plot_dir,
                                          y_min_line=UAV_MIN_ENERGY,
                                          y_min_label="最低安全能量")

            plotter.plot_metric_over_time(env.simulation_log, N_UAVS,
                                          metric_key="uav_tx_powers_mW",  # 使用新的日志键
                                          title="各UAV传输功率变化 (mW)",
                                          ylabel="传输功率 (mW)",
                                          output_dir=output_plot_dir,
                                          # 可选：如果你想在图上画出最大允许功率线
                                          y_min_line=None,  # 这里没有最小功率线，但可以设一个y_max_line (如果修改了plot_metric_over_time)
                                          # y_max_val = UAV_MAX_TRANS_POWER * 1000 # 设定Y轴上限
                                          )

            print(f"所有图表已尝试保存到 '{output_plot_dir}' 文件夹中。")
            # 如果需要立即显示图表，取消下面这行的注释
            # plotter.show_all_plots()

        except Exception as e:
            print(f"生成图表时发生错误: {e}")
            print("请确保 matplotlib 安装正确并且日志数据有效。")
    elif not plotter_exists:
        print("绘图模块 (plotter.py) 未能加载或 matplotlib 未安装，跳过图表生成。")
    elif not env.simulation_log:
        print("没有日志数据，无法生成图表。")


if __name__ == "__main__":
    main()