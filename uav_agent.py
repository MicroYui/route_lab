# -*- coding: utf-8 -*-
# uav_agent.py
import numpy as np
import math

import config
from config import *  # UAV_MIN_SPEED, MAX_COMMUNICATION_RANGE etc.
from decision_algorithms import LyapunovDecisionAlgorithm, GreedyMaxPowerAlgorithm, GreedyFixedPowerAlgorithm, \
    LoadBalancingGreedyAlgorithm
from utils import calculate_distance  # calculate_distance is now 2D
from mobility_models import GaussMarkovMobilityModel  # Expects 2D initial_pos
from communication_models import calculate_channel_gain  # Expects 2D positions
import os
import csv


class UAV:
    def __init__(self, uav_id, initial_pos_xy, data_center_pos_xy, sim_env):  # 接收二维参数
        self.id = uav_id
        self.sim_env = sim_env

        self.mobility_model = GaussMarkovMobilityModel(uav_id, initial_pos_xy)
        self.current_pos = list(initial_pos_xy[:2])  # 存储为二维列表
        self.current_velocity = [0.0, 0.0]  # 二维速度
        self.prev_pos = list(self.current_pos)  # 二维

        self.data_queue_bits = 0
        self.energy = float(UAV_MAX_ENERGY)
        self.virtual_energy_queue = 0.0
        self.P_avg = (float(UAV_MAX_ENERGY) - float(UAV_MIN_ENERGY)) / float(
            TOTAL_SIMULATION_TIME) if TOTAL_SIMULATION_TIME > 0 else 0

        self.neighbors = {}
        self.link_durations_L_D = {}
        self.LHT = float('inf')
        self.hello_interval = 1.0
        self.time_since_last_hello = self.hello_interval

        self.data_center_pos = list(data_center_pos_xy[:2])  # 存储二维

        self.current_tx_power_W = 0.0
        self.chosen_next_hop_id = None
        self.chosen_tx_power_to_next_hop_W = 0.0
        self.last_successful_tx_rate_bps = 0.0

        if config.DECISION_ALGORITHM == "lyapunov":
            self.decision_algorithm = LyapunovDecisionAlgorithm(self)
        elif config.DECISION_ALGORITHM == "greedy_max_power":
            self.decision_algorithm = GreedyMaxPowerAlgorithm(self)
        elif config.DECISION_ALGORITHM == "greedy_fixed_power":
            self.decision_algorithm = GreedyFixedPowerAlgorithm(self)
        elif config.DECISION_ALGORITHM == "load_balancing":
            self.decision_algorithm = LoadBalancingGreedyAlgorithm(self)
        else:
            raise ValueError(f"未知的决策算法类型: {config.DECISION_ALGORITHM}")

        self.debug_file_path = f"debug_uav_{uav_id}.csv"

    def get_position(self):
        """返回UAV的当前二维位置。"""
        return self.current_pos  # 返回二维列表

    def update_state_at_start_of_step(self, current_time):
        self.prev_pos = list(self.current_pos)
        # mobility_model.update() 现在返回二维 pos 和 vel
        new_pos_list, new_vel_list = self.mobility_model.update()
        self.current_pos = new_pos_list[:2]  # 确保是二维
        self.current_velocity = new_vel_list[:2]  # 确保是二维

        self.time_since_last_hello += DELTA_T
        # if self.id < N_UAVS / 4:
        if self.id in DATA_SOURCE_UAV_IDS:
            self.generate_data(current_time)

    def generate_data(self, current_time):  # 逻辑不变
        num_packets_generated = np.random.poisson(DATA_GENERATION_RATE_AVG * DELTA_T)
        generated_bits = num_packets_generated * DATA_PACKET_SIZE_BITS
        if self.data_queue_bits + generated_bits <= UAV_MAX_QUEUE_LENGTH * DATA_PACKET_SIZE_BITS:
            self.data_queue_bits += generated_bits
        else:
            self.data_queue_bits = UAV_MAX_QUEUE_LENGTH * DATA_PACKET_SIZE_BITS

    def _calculate_link_duration_L_D(self, neighbor_id, neighbor_pos_xy, neighbor_velocity_vector_xy):
        """ 根据建模文档计算 L_D(i,j) - 使用二维数据 """
        R_comm = float(MAX_COMMUNICATION_RANGE)
        V_min_abs = float(UAV_MIN_SPEED) if UAV_MIN_SPEED > 0 else 1.0

        T_max_link = R_comm / V_min_abs

        neighbor_prev_pos_xy = self.neighbors.get(neighbor_id, {}).get('pos_prev', neighbor_pos_xy)
        d_ij_t1 = calculate_distance(self.prev_pos, neighbor_prev_pos_xy)  # 使用二维距离
        d_ij_t2 = calculate_distance(self.current_pos, neighbor_pos_xy)  # 使用二维距离

        delta_d = d_ij_t2 - d_ij_t1
        time_diff = float(DELTA_T)

        if delta_d > 0:  # 距离增大
            relative_speed_scalar = delta_d / time_diff if time_diff > 0 else V_min_abs
            if relative_speed_scalar <= 1e-6:
                L_D = T_max_link
            else:
                L_D = (R_comm - d_ij_t2) / relative_speed_scalar
            return max(0, L_D)
        else:  # 距离减小或不变
            if R_comm <= 1e-6: return T_max_link  # 避免除以0
            L_D = (d_ij_t2 / R_comm) * T_max_link
            return min(L_D, T_max_link)

    def update_hello_params(self):  # 逻辑不变，依赖的 L_D 计算已改为2D
        if not self.link_durations_L_D:
            self.LHT = float(MAX_COMMUNICATION_RANGE) / (float(UAV_MIN_SPEED) if UAV_MIN_SPEED > 0 else 1.0)
        else:
            valid_durations = [d for d in self.link_durations_L_D.values() if d > 0]
            self.LHT = min(valid_durations) if valid_durations else (
                        float(MAX_COMMUNICATION_RANGE) / (float(UAV_MIN_SPEED) if UAV_MIN_SPEED > 0 else 1.0))
        if self.LHT <= 1e-3: self.LHT = 0.1
        self.hello_interval = TAU_HELLO * self.LHT
        self.hello_interval = max(0.2 * DELTA_T, min(self.hello_interval, 5.0 * DELTA_T))

    def create_hello_message(self, current_time):
        if self.time_since_last_hello >= self.hello_interval or (
                self.LHT < 2 * DELTA_T and self.time_since_last_hello >= DELTA_T) or not self.neighbors:
            self.time_since_last_hello = 0.0
            energy_diff = float(UAV_MAX_ENERGY - UAV_MIN_ENERGY)
            if energy_diff <= 0: energy_diff = 1.0
            normalized_energy = (self.energy - float(UAV_MIN_ENERGY)) / energy_diff

            msg = {
                "uav_id": self.id, "timestamp": current_time,
                "position": list(self.current_pos[:2]),  # 发送二维位置
                "velocity": list(self.current_velocity[:2]),  # 发送二维速度
                "energy_remaining": self.energy, "normalized_energy": np.clip(normalized_energy, 0, 1),
                "queue_length_bits": self.data_queue_bits, "virtual_queue_length": self.virtual_energy_queue,
            }
            return msg
        return None

    def process_hello_message(self, hello_msg, current_time):
        neighbor_id = hello_msg["uav_id"]
        if neighbor_id == self.id: return

        neighbor_pos_xy = hello_msg["position"][:2]  # 确保是二维
        neighbor_vel_xy = hello_msg["velocity"][:2]  # 确保是二维

        neighbor_data_before_update = self.neighbors.get(neighbor_id, {})
        prev_neighbor_pos_for_LD = neighbor_data_before_update.get('position', neighbor_pos_xy)

        self.neighbors[neighbor_id] = {
            "id": neighbor_id, "timestamp": hello_msg["timestamp"],
            "position": neighbor_pos_xy,  # 存储二维
            "pos_prev": prev_neighbor_pos_for_LD[:2],  # 存储二维
            "velocity": neighbor_vel_xy,  # 存储二维
            "energy_remaining": hello_msg["energy_remaining"],
            "normalized_energy": hello_msg["normalized_energy"],
            "queue_length_bits": hello_msg["queue_length_bits"],
            "virtual_queue_length": hello_msg["virtual_queue_length"],
            "last_seen_time": current_time
        }
        self.link_durations_L_D[neighbor_id] = self._calculate_link_duration_L_D(
            neighbor_id, neighbor_pos_xy, neighbor_vel_xy
        )
        self.update_hello_params()

    def cleanup_neighbors(self, current_time, expiry_multiplier=3.0):  # 逻辑不变
        fixed_expiry_timeout = 5.0 * DELTA_T
        expired_neighbors = [nid for nid, data in self.neighbors.items() if
                             current_time - data.get("last_seen_time", -float('inf')) > fixed_expiry_timeout]
        for nid in expired_neighbors:
            del self.neighbors[nid]
            if nid in self.link_durations_L_D: del self.link_durations_L_D[nid]
        if expired_neighbors: self.update_hello_params()

    def select_next_hop_and_power(self, current_time, estimated_interference_at_potential_rx_nodes):
        chosen_hop, chosen_power = self.decision_algorithm.select_next_hop_and_power(
            current_time,
            estimated_interference_at_potential_rx_nodes
        )

        # 更新UAV自身的决策结果变量
        self.chosen_next_hop_id = chosen_hop
        self.chosen_tx_power_to_next_hop_W = chosen_power

        # current_tx_power_W 反映了本时间步如果传输，将使用的总功率
        # 对于单播，它就等于 chosen_tx_power_to_next_hop_W
        if chosen_hop is not None and chosen_power > 1e-9:
            self.current_tx_power_W = chosen_power
        else:
            self.current_tx_power_W = 0.0
            # 确保如果决策是不发送，这些值也明确为None/0
            self.chosen_next_hop_id = None
            self.chosen_tx_power_to_next_hop_W = 0.0
        return chosen_hop, chosen_power

    def update_queues_and_energy_post_tx(self, bits_transmitted_successfully, bits_received, current_time):  # 逻辑不变
        current_q_val_before_tx_rx = self.data_queue_bits
        q_after_tx = max(0.0, current_q_val_before_tx_rx - bits_transmitted_successfully)
        q_after_rx_and_prev_gen = q_after_tx + bits_received
        self.data_queue_bits = min(q_after_rx_and_prev_gen, float(UAV_MAX_QUEUE_LENGTH * DATA_PACKET_SIZE_BITS))

        energy_consumed_Joules = self.current_tx_power_W * float(DELTA_T)
        self.energy -= energy_consumed_Joules
        self.energy = max(0.0, self.energy)
        if self.energy < float(UAV_MIN_ENERGY) and DEBUG_LEVEL >= 0:
            print(
                f"警告: UAV {self.id} 电量 {self.energy:.2f} 低于最低安全阈值 {UAV_MIN_ENERGY:.2f} (在时间 {current_time:.2f}s)!")

        accumulated_change = self.current_tx_power_W * float(DELTA_T) - self.P_avg * float(DELTA_T)
        self.virtual_energy_queue = max(0.0, self.virtual_energy_queue + accumulated_change)

    def _write_debug_to_csv(self, row_data):
        """追加写入CSV文件"""
        try:
            with open(self.debug_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            print(f"写入CSV文件失败: {e}")