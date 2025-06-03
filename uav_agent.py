# -*- coding: utf-8 -*-
# uav_agent.py
import numpy as np
import math
from config import *  # UAV_MIN_SPEED, MAX_COMMUNICATION_RANGE etc.
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

        self.debug_file_path = f"debug_uav_{uav_id}.csv"
        # 初始化CSV文件头
        with open(self.debug_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'uav_id', 'target_id', 'Q_i_t', 'D_i_t', 
                'Psi_i_t', 'C_i_t', 'P_star', 'N_ij', 'h_ij', 
                'correction_term', 'P_ij_trans_opt'
            ])

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
        self.chosen_next_hop_id = None
        self.chosen_tx_power_to_next_hop_W = 0.0
        best_W_ij_EG = -float('inf')

        if self.data_queue_bits < DATA_PACKET_SIZE_BITS:
            self.current_tx_power_W = 0.0
            return None, 0.0

        energy_denom = max(self.energy - float(UAV_MIN_ENERGY), 0.0) + float(LYAPUNOV_EPSILON)
        Psi_i_t = ((float(UAV_MAX_ENERGY) - float(UAV_MIN_ENERGY)) ** 2 / (energy_denom ** 3))
        C_i_t = float(LYAPUNOV_BETA) * Psi_i_t + float(LYAPUNOV_GAMMA) * self.virtual_energy_queue
        Q_i_t_for_D = self.data_queue_bits
        D_i_t = Q_i_t_for_D * float(DELTA_T) + float(LYAPUNOV_V)

        potential_next_hops = []
        for neighbor_id, neighbor_data in self.neighbors.items():
            dist_to_neighbor = calculate_distance(self.current_pos, neighbor_data["position"])  # 2D distance
            if dist_to_neighbor <= MAX_COMMUNICATION_RANGE:
                potential_next_hops.append({
                    "id": neighbor_id, "pos": neighbor_data["position"][:2],  # 使用二维位置
                    "norm_energy": neighbor_data["normalized_energy"], "is_dc": False
                })

        dist_to_dc = calculate_distance(self.current_pos, self.data_center_pos)  # 2D distance
        if dist_to_dc <= MAX_COMMUNICATION_RANGE:
            potential_next_hops.append({
                "id": "D", "pos": self.data_center_pos[:2],  # 使用二维位置
                "norm_energy": 1.0, "is_dc": True
            })

        if not potential_next_hops:
            self.current_tx_power_W = 0.0
            return None, 0.0

        for hop_candidate in potential_next_hops:
            j_id = hop_candidate["id"]
            j_pos_xy = hop_candidate["pos"]  # 已经是二维

            h_ij = calculate_channel_gain(self.current_pos, j_pos_xy)  # 使用二维位置
            if h_ij <= 1e-12: continue

            estimated_I_ij = estimated_interference_at_potential_rx_nodes.get(j_id, 0.0)
            N_ij = estimated_I_ij + float(NOISE_POWER)
            if N_ij <= 1e-18: N_ij = 1e-18

            numerator_P_star = D_i_t * float(BANDWIDTH)
            denominator_P_star = C_i_t * float(DELTA_T) * math.log(2)
            if abs(denominator_P_star) < 1e-9 or abs(h_ij) < 1e-9:
                P_ij_trans_star = float(UAV_MAX_TRANS_POWER) if denominator_P_star > 0 else 0.0
            else:
                P_ij_trans_star = (numerator_P_star / denominator_P_star) - (N_ij / h_ij)
            P_ij_trans_opt = min(max(0.0, P_ij_trans_star), float(UAV_MAX_TRANS_POWER))

            if self.id == 1 and (P_ij_trans_opt > 1e-9 or int(current_time) % 10 == 0):
                # 直接追加写入CSV
                self._write_debug_to_csv([
                    current_time, self.id, j_id, Q_i_t_for_D, D_i_t,
                    Psi_i_t, C_i_t,
                    numerator_P_star/denominator_P_star if abs(denominator_P_star) > 1e-9 else 0,
                    N_ij, h_ij, N_ij / h_ij, P_ij_trans_opt
                ])

            r_ij_opt_bps = 0.0
            if P_ij_trans_opt > 1e-9:
                sinr_opt = (h_ij * P_ij_trans_opt) / N_ij
                if sinr_opt > 0:
                    r_ij_opt_bps = float(BANDWIDTH) * math.log2(1 + sinr_opt)

            W_ij = D_i_t * r_ij_opt_bps - C_i_t * float(DELTA_T) * P_ij_trans_opt

            W_ij_EG = -float('inf')
            dist_i_to_D = calculate_distance(self.current_pos, self.data_center_pos)  # 2D distance

            if hop_candidate["is_dc"]:
                if dist_i_to_D > 1e-3: W_ij_EG = W_ij
            else:
                dist_j_to_D = calculate_distance(j_pos_xy, self.data_center_pos)  # 2D distance
                if dist_i_to_D > 1e-3:
                    D_prog_ij = (dist_i_to_D - dist_j_to_D) / dist_i_to_D
                    if D_prog_ij > 0:
                        E_tilde_j = hop_candidate["norm_energy"]
                        W_ij_EG = W_ij * E_tilde_j

            if W_ij_EG > best_W_ij_EG:
                best_W_ij_EG = W_ij_EG
                self.chosen_next_hop_id = j_id
                self.chosen_tx_power_to_next_hop_W = P_ij_trans_opt

        if self.chosen_next_hop_id and self.chosen_tx_power_to_next_hop_W > 1e-9:
            self.current_tx_power_W = self.chosen_tx_power_to_next_hop_W
            return self.chosen_next_hop_id, self.chosen_tx_power_to_next_hop_W
        else:
            self.chosen_next_hop_id = None
            self.chosen_tx_power_to_next_hop_W = 0.0
            self.current_tx_power_W = 0.0
            return None, 0.0

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