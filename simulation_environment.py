# -*- coding: utf-8 -*-
# simulation_environment.py
import numpy as np
import random
from config import *  # AREA_WIDTH, AREA_HEIGHT, DATA_CENTER_POSITION (now 2D)
from uav_agent import UAV  # Expects 2D initial_pos_xy, data_center_pos_xy
from data_center_node import DataCenter  # Expects 2D position
from communication_models import calculate_channel_gain, calculate_sinr, calculate_transmission_rate
from utils import calculate_distance


class SimulationEnvironment:
    def __init__(self, num_uavs):
        self.num_uavs = num_uavs
        self.uavs = []
        # DataCenter now initializes with a 2D position from config
        self.data_center = DataCenter(dc_id="D")
        self.current_time = 0.0
        self.simulation_log = []

        self.current_step_intended_transmissions = {}
        self.interference_estimation_for_decision = {i: 0.0 for i in range(num_uavs)}
        self.interference_estimation_for_decision["D"] = 0.0

        if USE_MANUAL_UAV_POSITIONS:
            if UAV_INITIAL_POSITIONS is None or len(UAV_INITIAL_POSITIONS) != self.num_uavs:
                print(f"错误: USE_MANUAL_UAV_POSITIONS 为 True, 但 UAV_INITIAL_POSITIONS "
                      f"未定义或其长度 ({len(UAV_INITIAL_POSITIONS) if UAV_INITIAL_POSITIONS else 'None'}) "
                      f"与 N_UAVS ({self.num_uavs}) 不匹配。请检查 config.py。")
                print("仿真将退出。")
                exit()  # 或者抛出异常
            print(f"信息: 正在使用 config.py 中手动设定的 {self.num_uavs} 个无人机初始位置。")
            initial_positions_to_use = UAV_INITIAL_POSITIONS
        else:
            print(f"信息: USE_MANUAL_UAV_POSITIONS 为 False。将为 {self.num_uavs} 个无人机随机生成初始位置。")
            initial_positions_to_use = []
            temp_positions_set = set()  # 用于确保随机位置不完全重合
            for _ in range(self.num_uavs):
                while True:
                    pos_candidate = (
                        random.uniform(0, AREA_WIDTH),
                        random.uniform(0, AREA_HEIGHT)
                    )
                    if pos_candidate not in temp_positions_set:
                        temp_positions_set.add(pos_candidate)
                        initial_positions_to_use.append(pos_candidate)
                        break

        for i in range(self.num_uavs):
            initial_pos_xy = initial_positions_to_use[i]

            uav = UAV(uav_id=i,
                      initial_pos_xy=list(initial_pos_xy),
                      data_center_pos_xy=self.data_center.get_position(),
                      sim_env=self)
            self.uavs.append(uav)
            if DEBUG_LEVEL >= 1:
                print(f"UAV {i} 初始化于位置: {initial_pos_xy}")

    def run_step(self):
        # ... (大部分 run_step 逻辑不变, 因为内部 UAV 和 DC 对象现在处理二维坐标) ...
        # 例如，在 Hello 消息交换中:
        # distance_to_sender = calculate_distance(uav_receiver.current_pos, hello_msg["position"])
        # uav_receiver.current_pos 和 hello_msg["position"] 都应该是二维的

        # 在处理传输和干扰计算时:
        # tx_uav.current_pos 和 rx_node_object.get_position() 都是二维的
        # calculate_channel_gain 会正确使用二维位置
        if DEBUG_LEVEL >= 1 and int(self.current_time) % (DELTA_T * 10) == 0:
            print(f"\n--- 时间步: {self.current_time:.2f}s ---")

        hello_messages_to_broadcast_this_step = []
        for uav in self.uavs:
            if uav.energy < UAV_MIN_ENERGY * 0.5: continue  # 更严格的停止条件
            uav.update_state_at_start_of_step(self.current_time)
            hello_msg = uav.create_hello_message(self.current_time)
            if hello_msg:
                hello_messages_to_broadcast_this_step.append(hello_msg)

        for uav_receiver in self.uavs:
            if uav_receiver.energy < UAV_MIN_ENERGY * 0.5: continue
            uav_receiver.cleanup_neighbors(self.current_time)
            for hello_msg in hello_messages_to_broadcast_this_step:
                if uav_receiver.id == hello_msg["uav_id"]: continue
                # hello_msg["position"] 已经是二维的了
                distance_to_sender = calculate_distance(uav_receiver.get_position(), hello_msg["position"])
                if distance_to_sender <= MAX_COMMUNICATION_RANGE:
                    uav_receiver.process_hello_message(hello_msg, self.current_time)

        self.current_step_intended_transmissions.clear()
        for uav in self.uavs:
            if uav.energy < UAV_MIN_ENERGY * 0.5: continue
            chosen_next_hop_id, chosen_tx_power_W = uav.select_next_hop_and_power(
                self.current_time, self.interference_estimation_for_decision
            )
            if chosen_next_hop_id is not None and chosen_tx_power_W > 1e-9:
                self.current_step_intended_transmissions[(uav.id, chosen_next_hop_id)] = chosen_tx_power_W

        actual_bits_received_by_node_map = {uav.id: 0.0 for uav in self.uavs}
        actual_bits_received_by_node_map["D"] = 0.0
        actual_bits_transmitted_by_uav_map = {uav.id: 0.0 for uav in self.uavs}

        new_interference_estimation = {nid: 0.0 for nid in self.interference_estimation_for_decision.keys()}

        # 收集所有节点的当前二维位置
        all_nodes_current_pos_xy = {uav.id: uav.get_position() for uav in self.uavs}
        all_nodes_current_pos_xy["D"] = self.data_center.get_position()

        for (tx_uav_id, rx_node_id), tx_power_W in self.current_step_intended_transmissions.items():
            tx_uav = self.uavs[tx_uav_id]
            rx_node_object_pos_xy = all_nodes_current_pos_xy.get(rx_node_id)
            if rx_node_object_pos_xy is None: continue

            channel_gain_h_ij = calculate_channel_gain(tx_uav.get_position(), rx_node_object_pos_xy)
            if channel_gain_h_ij <= 1e-12:
                tx_uav.last_successful_tx_rate_bps = 0.0
                continue

            current_link_interference_W = 0.0
            for (other_tx_id, _), other_tx_power_W in self.current_step_intended_transmissions.items():
                if other_tx_id == tx_uav_id: continue

                other_tx_uav_pos_xy = all_nodes_current_pos_xy.get(other_tx_id)
                if other_tx_uav_pos_xy is None: continue

                channel_gain_h_kj_to_rx = calculate_channel_gain(other_tx_uav_pos_xy, rx_node_object_pos_xy)
                current_link_interference_W += channel_gain_h_kj_to_rx * other_tx_power_W

            actual_sinr_on_link = calculate_sinr(tx_power_W, channel_gain_h_ij, current_link_interference_W)
            actual_rate_bps_on_link = calculate_transmission_rate(actual_sinr_on_link)

            bits_can_be_transmitted_in_slot = actual_rate_bps_on_link * float(DELTA_T)
            bits_to_actually_transmit = min(tx_uav.data_queue_bits, bits_can_be_transmitted_in_slot)

            if actual_rate_bps_on_link > 0 and bits_to_actually_transmit >= 1:
                tx_uav.last_successful_tx_rate_bps = actual_rate_bps_on_link
                actual_bits_transmitted_by_uav_map[tx_uav_id] += bits_to_actually_transmit
                if rx_node_id == "D":
                    self.data_center.receive_data(tx_uav_id, bits_to_actually_transmit, self.current_time)
                else:  # rx_node_id 必须是 int 类型的 UAV ID
                    if isinstance(rx_node_id, int):
                        actual_bits_received_by_node_map[rx_node_id] += bits_to_actually_transmit
            else:
                tx_uav.last_successful_tx_rate_bps = 0.0

        # 更新干扰估计 (对每个节点 J，计算所有其他发射 K 对 J 的总干扰)
        for node_j_id in new_interference_estimation.keys():
            pos_j_xy = all_nodes_current_pos_xy.get(node_j_id)
            if pos_j_xy is None: continue

            total_interference_at_j = 0.0
            for (tx_k_id, _), tx_power_k_W in self.current_step_intended_transmissions.items():
                # 干扰源 k 不能是接收节点 j 自己
                # 并且，我们计算的是 tx_k_id 对 node_j_id 的干扰
                if tx_k_id == node_j_id: continue

                pos_k_xy = all_nodes_current_pos_xy.get(tx_k_id)
                if pos_k_xy is None: continue

                h_kj = calculate_channel_gain(pos_k_xy, pos_j_xy)
                total_interference_at_j += h_kj * tx_power_k_W
            new_interference_estimation[node_j_id] = total_interference_at_j
        self.interference_estimation_for_decision = new_interference_estimation

        for i, uav in enumerate(self.uavs):
            if uav.energy < UAV_MIN_ENERGY * 0.5: continue
            uav.update_queues_and_energy_post_tx(
                actual_bits_transmitted_by_uav_map[uav.id],
                actual_bits_received_by_node_map.get(uav.id, 0.0),  # 使用 .get 以防万一
                self.current_time
            )

        self._log_simulation_metrics()  # log_simulation_metrics 也需要适应二维
        self.current_time += float(DELTA_T)
        active_uav_exists = any(uav.energy >= UAV_MIN_ENERGY for uav in self.uavs)
        return self.current_time < TOTAL_SIMULATION_TIME and active_uav_exists

    def _log_simulation_metrics(self):
        # ... (与上一版类似，但 uav_positions 现在记录的是二维位置) ...
        current_uav_positions = {}  # 将存储二维位置
        current_uav_queues_kb = {}
        current_uav_energies = {}
        current_uav_tx_powers_mW = {}

        for uav in self.uavs:
            current_uav_positions[uav.id] = list(uav.get_position()[:2])  # 确保记录二维
            current_uav_queues_kb[uav.id] = uav.data_queue_bits / (8.0 * 1024.0)
            current_uav_energies[uav.id] = uav.energy
            current_uav_tx_powers_mW[uav.id] = uav.current_tx_power_W * 1000.0  # 转换为毫瓦

        # ... (其余的聚合指标计算不变) ...
        avg_total_throughput_bps = self.data_center.get_total_throughput_bps(
            self.current_time if self.current_time > 0 else float(DELTA_T))
        active_energies_normalized = [uav.energy / (float(UAV_MAX_ENERGY) if UAV_MAX_ENERGY > 0 else 1.0) for uav in
                                      self.uavs if uav.energy >= UAV_MIN_ENERGY]
        energy_balance_std_dev = np.std(active_energies_normalized) if active_energies_normalized else 0.0
        min_energy_among_active = min([uav.energy for uav in self.uavs if uav.energy >= UAV_MIN_ENERGY], default=0.0)
        first_uav_below_threshold_time = -1.0
        # (此处的 first_dead_time_recorded 逻辑保持不变)
        if not hasattr(self, '_first_dead_time_recorded'):
            if any(uav.energy < UAV_MIN_ENERGY for uav in self.uavs):
                first_uav_below_threshold_time = self.current_time;
                self._first_dead_time_recorded = True
        elif hasattr(self, '_first_dead_time_recorded') and self._first_dead_time_recorded:
            if self.simulation_log and self.simulation_log[-1].get("首个UAV低电量时间 (s)", -1.0) != -1.0:
                first_uav_below_threshold_time = self.simulation_log[-1]["首个UAV低电量时间 (s)"]
            elif any(uav.energy < UAV_MIN_ENERGY for uav in self.uavs) and (
                    not self.simulation_log or self.simulation_log[-1].get("首个UAV低电量时间 (s)", -1.0) == -1.0):
                first_uav_below_threshold_time = self.current_time

        num_active_uavs = sum(1 for uav in self.uavs if uav.energy >= UAV_MIN_ENERGY)
        active_q_list = [current_uav_queues_kb[uav.id] for uav in self.uavs if
                         uav.energy >= UAV_MIN_ENERGY and uav.id in current_uav_queues_kb]
        avg_queue_len_KB = np.mean(active_q_list) if active_q_list else 0.0

        log_entry = {
            "时间 (s)": self.current_time,
            "网络吞吐量 (Kbps)": avg_total_throughput_bps / 1024.0,
            "能量均衡度 (归一化标准差)": energy_balance_std_dev,
            "最低剩余能量 (J)": min_energy_among_active,
            "首个UAV低电量时间 (s)": first_uav_below_threshold_time,
            "活动UAV数量": num_active_uavs,
            "平均队列长度 (KB)": avg_queue_len_KB,
            "数据中心接收总比特": self.data_center.total_data_received_bits,
            "uav_positions": current_uav_positions,  # 包含二维位置
            "uav_queues_kb": current_uav_queues_kb,
            "uav_energies": current_uav_energies,
            "uav_tx_powers_mW": current_uav_tx_powers_mW
        }
        self.simulation_log.append(log_entry)

        if DEBUG_LEVEL >= 1 and int(self.current_time) % (DELTA_T * 10) == 0:
            active_tx_powers_this_step = [pwr for pwr in current_uav_tx_powers_mW.values() if pwr > 0]
            avg_active_tx_pwr = np.mean(active_tx_powers_this_step) if active_tx_powers_this_step else 0
            print(
                f"指标@{self.current_time:.0f}s: 吞吐量 {log_entry['网络吞吐量 (Kbps)']:.2f} Kbps, 活动UAV数 {log_entry['活动UAV数量']}, "
                f"平均发射功率(激活时) {avg_active_tx_pwr:.2f} mW")