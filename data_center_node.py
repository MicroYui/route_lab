# -*- coding: utf-8 -*-
# data_center_node.py
from config import DATA_CENTER_POSITION, DELTA_T # DATA_CENTER_POSITION 现在是二维的

class DataCenter:
    def __init__(self, dc_id="D"):
        self.id = dc_id
        self.position = list(DATA_CENTER_POSITION[:2]) # 存储为二维列表
        self.total_data_received_bits = 0
        self.log = []

    def receive_data(self, uav_source_id, data_amount_bits, current_time):
        self.total_data_received_bits += data_amount_bits
        self.log.append({
            "time": current_time,
            "source_uav": uav_source_id,
            "bits": data_amount_bits
        })

    def get_throughput_bps(self, time_window_start, time_window_end):
        bits_in_window = 0
        for entry in self.log:
            if time_window_start <= entry["time"] < time_window_end:
                bits_in_window += entry["bits"]
        duration = time_window_end - time_window_start
        if duration <= 0: return 0.0
        return bits_in_window / duration

    def get_total_throughput_bps(self, total_simulation_time):
        if total_simulation_time <= 0: return 0.0
        return self.total_data_received_bits / total_simulation_time

    def get_position(self):
        """返回数据中心的二维位置。"""
        return self.position # 返回二维列表