import math
import numpy as np
import config  # 导入config以获取全局参数和算法特定参数
from utils import calculate_distance
from communication_models import calculate_channel_gain, calculate_sinr, calculate_transmission_rate
import csv


class BaseDecisionAlgorithm:
    """决策算法的基类 (接口定义)"""

    def __init__(self, uav_agent):
        self.uav = uav_agent  # 持有UAV代理的引用，以便访问其状态

    def select_next_hop_and_power(self, current_time, estimated_interference_at_potential_rx_nodes):
        """
        选择下一跳和传输功率。

        参数:
            current_time (float): 当前仿真时间。
            estimated_interference_at_potential_rx_nodes (dict): 
                {node_id: interference_power_W} 对潜在接收节点处总干扰的估计。

        返回:
            tuple: (chosen_next_hop_id, chosen_tx_power_W)
                   chosen_next_hop_id 可以是邻居ID、"D" (数据中心) 或 None。
                   chosen_tx_power_W 是选择的传输功率 (瓦特)。
        """
        raise NotImplementedError("子类必须实现此方法")


class LyapunovDecisionAlgorithm(BaseDecisionAlgorithm):

    def __init__(self, uav_agent):
        super().__init__(uav_agent)
        # 李雅普诺夫特定参数可以从config中获取，或在此处硬编码/传递
        self.V = float(config.LYAPUNOV_V)
        self.beta = float(config.LYAPUNOV_BETA)
        self.gamma = float(config.LYAPUNOV_GAMMA)
        self.epsilon = float(config.LYAPUNOV_EPSILON)
        self.debug_file_path = f"debug_uav_{uav_agent.id}.csv"
        with open(self.debug_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'uav_id', 'target_id', 'Q_i_t', 'D_i_t',
                'Psi_i_t', 'C_i_t', 'beta_Psi_i_t', 'gamma_F_t', 'up', 'down', 'P_star', 'N_ij', 'h_ij',
                'correction_term', 'P_ij_trans_opt'
            ])

    def select_next_hop_and_power(self, current_time, estimated_interference_at_potential_rx_nodes):
        uav = self.uav  # 获取UAV对象引用

        chosen_next_hop_id = None
        chosen_tx_power_to_next_hop_W = 0.0
        best_W_ij_EG = -float('inf')

        if uav.data_queue_bits < config.DATA_PACKET_SIZE_BITS:
            return None, 0.0

        # energy_denom = max(uav.energy - float(config.UAV_MIN_ENERGY), 0.0) + self.epsilon
        # Psi_i_t = ((float(config.UAV_MAX_ENERGY) - float(config.UAV_MIN_ENERGY)) ** 2 / (energy_denom ** 3))

        Psi_i_t = float(20) - uav.energy
        beta_Psi_i_t = float(self.beta) * Psi_i_t
        gamma_F_i_t = float(self.gamma) * uav.virtual_energy_queue
        C_i_t = beta_Psi_i_t + gamma_F_i_t

        Q_i_t_for_D = uav.data_queue_bits
        D_i_t = Q_i_t_for_D * float(config.DELTA_T) + self.V

        potential_next_hops = []
        for neighbor_id, neighbor_data in uav.neighbors.items():
            dist_to_neighbor = calculate_distance(uav.current_pos, neighbor_data["position"])
            if dist_to_neighbor <= config.MAX_COMMUNICATION_RANGE:
                potential_next_hops.append({
                    "id": neighbor_id, "pos": neighbor_data["position"][:2],
                    "norm_energy": neighbor_data["normalized_energy"], "is_dc": False
                })
        dist_to_dc = calculate_distance(uav.current_pos, uav.data_center_pos)
        if dist_to_dc <= config.MAX_COMMUNICATION_RANGE:
            potential_next_hops.append({
                "id": "D", "pos": uav.data_center_pos[:2],
                "norm_energy": 1.0, "is_dc": True
            })

        if not potential_next_hops:
            return None, 0.0

        for hop_candidate in potential_next_hops:
            j_id = hop_candidate["id"]
            j_pos_xy = hop_candidate["pos"]

            h_ij = calculate_channel_gain(uav.current_pos, j_pos_xy)
            if h_ij <= 1e-12: continue

            estimated_I_ij = estimated_interference_at_potential_rx_nodes.get(j_id, 0.0)
            N_ij = estimated_I_ij + float(config.NOISE_POWER)
            if N_ij <= 1e-18: N_ij = 1e-18

            numerator_P_star = D_i_t * float(config.BANDWIDTH)
            denominator_P_star = C_i_t * float(config.DELTA_T) * math.log(2)

            if abs(denominator_P_star) < 1e-9 or abs(h_ij) < 1e-9:
                P_ij_trans_star = float(config.UAV_MAX_TRANS_POWER) if denominator_P_star > 0 else 0.0
            else:
                P_ij_trans_star = (numerator_P_star / denominator_P_star) - (N_ij / h_ij)
            P_ij_trans_opt = min(max(0.0, P_ij_trans_star), float(config.UAV_MAX_TRANS_POWER))

            if (P_ij_trans_opt > 1e-9 or int(current_time) % 10 == 0):
                try:
                    with open(self.debug_file_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            current_time, uav.id, j_id, Q_i_t_for_D, D_i_t,
                            Psi_i_t, C_i_t, beta_Psi_i_t, gamma_F_i_t, numerator_P_star, denominator_P_star,
                            numerator_P_star / denominator_P_star if abs(denominator_P_star) > 1e-9 else 0,
                            N_ij, h_ij, N_ij / h_ij, P_ij_trans_opt
                        ])
                except Exception as e:
                    print(f"写入CSV文件失败: {e}")

            r_ij_opt_bps = 0.0
            if P_ij_trans_opt > 1e-9:
                sinr_opt = (h_ij * P_ij_trans_opt) / N_ij
                if sinr_opt > 0:
                    r_ij_opt_bps = float(config.BANDWIDTH) * math.log2(1 + sinr_opt)

            W_ij = D_i_t * r_ij_opt_bps - C_i_t * float(config.DELTA_T) * P_ij_trans_opt
            W_ij_EG = -float('inf')
            dist_i_to_D = calculate_distance(uav.current_pos, uav.data_center_pos)
            if hop_candidate["is_dc"]:
                if dist_i_to_D > 1e-3: W_ij_EG = W_ij
            else:
                dist_j_to_D = calculate_distance(j_pos_xy, uav.data_center_pos)
                if dist_i_to_D > 1e-3:
                    D_prog_ij = (dist_i_to_D - dist_j_to_D) / dist_i_to_D
                    if D_prog_ij > 0:
                        E_tilde_j = hop_candidate["norm_energy"]
                        W_ij_EG = W_ij * E_tilde_j
            if W_ij_EG > best_W_ij_EG:
                best_W_ij_EG = W_ij_EG
                chosen_next_hop_id = j_id
                chosen_tx_power_to_next_hop_W = P_ij_trans_opt

        if chosen_next_hop_id and chosen_tx_power_to_next_hop_W > 1e-9:
            return chosen_next_hop_id, chosen_tx_power_to_next_hop_W
        else:
            return None, 0.0


class GreedyMaxPowerAlgorithm(BaseDecisionAlgorithm):
    """贪心算法：选择离数据中心最近的邻居，并使用最大功率传输"""

    def __init__(self, uav_agent):
        super().__init__(uav_agent)

    def select_next_hop_and_power(self, current_time, estimated_interference_at_potential_rx_nodes):
        uav = self.uav
        if uav.data_queue_bits < config.DATA_PACKET_SIZE_BITS:
            return None, 0.0

        best_next_hop_id = None
        min_dist_to_dc_for_next_hop = float('inf')
        current_pos_to_dc_dist = calculate_distance(uav.current_pos, uav.data_center_pos)

        potential_next_hops_for_greedy = []
        # 检查直接到数据中心
        if current_pos_to_dc_dist <= config.MAX_COMMUNICATION_RANGE:
            potential_next_hops_for_greedy.append({"id": "D", "pos": uav.data_center_pos, "dist_to_dc": 0.0})

        # 检查邻居
        for neighbor_id, neighbor_data in uav.neighbors.items():
            dist_to_neighbor = calculate_distance(uav.current_pos, neighbor_data["position"])
            if dist_to_neighbor <= config.MAX_COMMUNICATION_RANGE:
                # 选择能量健康的邻居，并且比当前UAV更接近数据中心
                if neighbor_data["normalized_energy"] > 0.1:  # 简单阈值，避免能量过低的
                    neighbor_pos_to_dc_dist = calculate_distance(neighbor_data["position"], uav.data_center_pos)
                    if neighbor_pos_to_dc_dist < current_pos_to_dc_dist:  # 必须是向D前进
                        potential_next_hops_for_greedy.append({
                            "id": neighbor_id,
                            "pos": neighbor_data["position"],
                            "dist_to_dc": neighbor_pos_to_dc_dist
                        })

        if not potential_next_hops_for_greedy:
            return None, 0.0

        # 从可选项中选择到DC距离最小的
        sorted_hops = sorted(potential_next_hops_for_greedy, key=lambda x: x["dist_to_dc"])
        best_next_hop_id = sorted_hops[0]["id"]

        if best_next_hop_id:
            # 贪心策略：使用最大允许功率
            return best_next_hop_id, float(config.UAV_MAX_TRANS_POWER)
        else:
            return None, 0.0


class GreedyFixedPowerAlgorithm(BaseDecisionAlgorithm):
    """贪心算法：选择离数据中心最近的邻居，并使用固定的“平均”功率传输"""

    def __init__(self, uav_agent):
        super().__init__(uav_agent)
        self.fixed_power = float(config.GREEDY_FIXED_TRANSMIT_POWER_W)  # 从config获取固定功率

    def select_next_hop_and_power(self, current_time, estimated_interference_at_potential_rx_nodes):
        uav = self.uav
        # 下一跳选择逻辑与 GreedyMaxPowerAlgorithm 相同
        if uav.data_queue_bits < config.DATA_PACKET_SIZE_BITS:
            return None, 0.0

        best_next_hop_id = None
        current_pos_to_dc_dist = calculate_distance(uav.current_pos, uav.data_center_pos)
        potential_next_hops_for_greedy = []
        if current_pos_to_dc_dist <= config.MAX_COMMUNICATION_RANGE:
            potential_next_hops_for_greedy.append({"id": "D", "pos": uav.data_center_pos, "dist_to_dc": 0.0})

        for neighbor_id, neighbor_data in uav.neighbors.items():
            dist_to_neighbor = calculate_distance(uav.current_pos, neighbor_data["position"])
            if dist_to_neighbor <= config.MAX_COMMUNICATION_RANGE:
                if neighbor_data["normalized_energy"] > 0.1:
                    neighbor_pos_to_dc_dist = calculate_distance(neighbor_data["position"], uav.data_center_pos)
                    if neighbor_pos_to_dc_dist < current_pos_to_dc_dist:
                        potential_next_hops_for_greedy.append({
                            "id": neighbor_id, "pos": neighbor_data["position"], "dist_to_dc": neighbor_pos_to_dc_dist
                        })

        if not potential_next_hops_for_greedy: return None, 0.0
        sorted_hops = sorted(potential_next_hops_for_greedy, key=lambda x: x["dist_to_dc"])
        best_next_hop_id = sorted_hops[0]["id"]

        if best_next_hop_id:
            # 贪心策略：使用固定的预设功率
            return best_next_hop_id, self.fixed_power
        else:
            return None, 0.0


class LoadBalancingGreedyAlgorithm(BaseDecisionAlgorithm):
    """负载均衡贪心算法：
       选择离数据中心更近且队列最短的邻居，并使用最大功率传输。
    """

    def __init__(self, uav_agent):
        super().__init__(uav_agent)

    def select_next_hop_and_power(self, current_time, estimated_interference_at_potential_rx_nodes):
        uav = self.uav
        if uav.data_queue_bits < config.DATA_PACKET_SIZE_BITS:
            return None, 0.0

        best_next_hop = None
        best_score = float('inf')  # 分数越小越好

        current_pos_to_dc_dist = calculate_distance(uav.current_pos, uav.data_center_pos)

        potential_next_hops = []
        # 检查直接到数据中心
        if current_pos_to_dc_dist <= config.MAX_COMMUNICATION_RANGE:
            # 给数据中心一个非常小的队列长度（负载），使其具有高优先级
            potential_next_hops.append({"id": "D", "pos": uav.data_center_pos, "queue_len": 0})

        # 检查邻居
        for neighbor_id, neighbor_data in uav.neighbors.items():
            dist_to_neighbor = calculate_distance(uav.current_pos, neighbor_data["position"])
            if dist_to_neighbor <= config.MAX_COMMUNICATION_RANGE:
                # 选择能量健康的邻居，并且比当前UAV更接近数据中心
                if neighbor_data["normalized_energy"] > 0.1:
                    neighbor_pos_to_dc_dist = calculate_distance(neighbor_data["position"], uav.data_center_pos)
                    if neighbor_pos_to_dc_dist < current_pos_to_dc_dist:
                        potential_next_hops.append({
                            "id": neighbor_id,
                            "pos": neighbor_data["position"],
                            "queue_len": neighbor_data["queue_length_bits"]  # 使用邻居的队列长度作为负载
                        })

        if not potential_next_hops:
            return None, 0.0

        # 从可选项中选择队列最短的作为下一跳
        # 如果队列长度相同，则选择ID较小的（或者可以进一步按距离排序）
        # 这里为了简单，直接选择队列最短的
        sorted_hops = sorted(potential_next_hops, key=lambda x: x["queue_len"])
        best_next_hop_id = sorted_hops[0]["id"]

        if best_next_hop_id:
            # 同样使用最大功率以简化对比
            return best_next_hop_id, float(config.UAV_MAX_TRANS_POWER)
        else:
            return None, 0.0