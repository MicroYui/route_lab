# -*- coding: utf-8 -*-
# run_experiments.py
import os
import time
import numpy as np
import json
import random
import pandas as pd  # 引入pandas方便数据处理
import config  # 导入配置，我们将动态修改它
from simulation_environment import SimulationEnvironment
import plotter  # 导入绘图模块


def run_single_simulation():
    """运行一次完整的仿真，并返回最终的性能指标。"""

    # 设定随机数种子以保证可复现性（如果需要）
    if config.SIMULATION_SEED is not None:
        random.seed(config.SIMULATION_SEED)
        np.random.seed(config.SIMULATION_SEED)

    # 初始化仿真环境
    env = SimulationEnvironment(num_uavs=config.N_UAVS)

    # 运行仿真
    simulation_is_active = True
    while simulation_is_active:
        simulation_is_active = env.run_step()
        if env.current_time >= config.TOTAL_SIMULATION_TIME:
            break
        if not any(uav.energy >= config.UAV_MIN_ENERGY for uav in env.uavs):
            break

    # --- 性能指标计算 ---
    if not env.simulation_log:
        print("警告: 仿真日志为空，无法计算性能指标。")
        return {"total_throughput_mb": 0, "energy_balance": -1, "network_lifetime": 0}

    # 1. 总吞吐量
    total_bits_to_dc = env.data_center.total_data_received_bits
    total_throughput_mb = total_bits_to_dc / (8 * 1024 * 1024)

    # 2. 能量均衡度
    final_normalized_energies = [(uav.energy / config.UAV_MAX_ENERGY) for uav in env.uavs]
    energy_balance = np.std(final_normalized_energies) if final_normalized_energies else -1

    # 3. 网络生存时间
    network_lifetime = env.current_time
    for log_item in env.simulation_log:
        first_dead_time = log_item.get("首个UAV低电量时间 (s)", -1.0)
        if first_dead_time != -1.0:
            network_lifetime = first_dead_time
            break

    return {
        "total_throughput_mb": total_throughput_mb,
        "energy_balance": energy_balance,
        "network_lifetime": network_lifetime
    }


def main():
    """主实验框架"""
    # 定义实验参数矩阵
    UAV_COUNTS_TO_TEST = [10, 20, 30, 40, 50]  # 要测试的无人机数量列表
    ALGORITHMS_TO_TEST = ["lyapunov", "greedy_max_power", "load_balancing"]  # 要测试的算法列表

    all_results = []

    # 创建唯一的输出目录
    output_dir_base = f"experiment_results_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir_base, exist_ok=True)

    # 循环遍历所有实验组合
    for num_uavs in UAV_COUNTS_TO_TEST:
        for algorithm in ALGORITHMS_TO_TEST:
            print(f"\n{'=' * 20} 正在运行实验 {'=' * 20}")
            print(f"无人机数量: {num_uavs}, 算法: {algorithm}")

            # 动态修改config模块中的参数
            config.N_UAVS = num_uavs
            config.DECISION_ALGORITHM = algorithm
            # 为了公平对比不同数量无人机的场景，使用随机位置生成
            config.USE_MANUAL_UAV_POSITIONS = False

            # 运行单次仿真并获取结果
            start_time = time.time()
            kpis = run_single_simulation()
            end_time = time.time()

            print(f"实验完成，耗时: {end_time - start_time:.2f} 秒")
            print(f"结果: {kpis}")

            # 存储本次运行的结果
            run_result = {
                "num_uavs": num_uavs,
                "algorithm": algorithm,
                **kpis  # 合并KPI字典
            }
            all_results.append(run_result)

    # --- 实验结束后，保存所有结果并绘图 ---
    if not all_results:
        print("没有实验结果可供分析。")
        return

    # 将结果保存到CSV文件
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_dir_base, "comparative_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n所有实验结果已保存到: {results_csv_path}")

    # 绘制对比图
    print("正在生成对比图表...")
    # 吞吐量对比
    plotter.plot_comparison(
        results_df,
        kpi_key="total_throughput_mb",
        title="不同算法下的网络总吞吐量对比",
        ylabel="总吞吐量 (MB)",
        output_dir=output_dir_base
    )
    # 能量均衡度对比
    plotter.plot_comparison(
        results_df,
        kpi_key="energy_balance",
        title="不同算法下的能量均衡度对比",
        ylabel="能量均衡度 (归一化标准差，越小越好)",
        output_dir=output_dir_base
    )
    # 网络生存时间对比
    plotter.plot_comparison(
        results_df,
        kpi_key="network_lifetime",
        title="不同算法下的网络生存时间对比",
        ylabel="网络生存时间 (秒)",
        output_dir=output_dir_base
    )
    print(f"所有对比图表已保存到 '{output_dir_base}' 文件夹中。")


if __name__ == "__main__":
    main()