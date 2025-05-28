# -*- coding: utf-8 -*-
# mobility_models.py
import numpy as np
import math
import random
from config import (AREA_WIDTH, AREA_HEIGHT, GM_ALPHA_SPEED, GM_MEAN_SPEED,
                    GM_STD_SPEED, GM_ALPHA_ANGLE, GM_STD_ANGLE_CHANGE, DELTA_T,
                    UAV_MAX_SPEED, UAV_MIN_SPEED)


class GaussMarkovMobilityModel:
    def __init__(self, uav_id, initial_pos_xy, initial_vel_vector_xy=None):
        self.uav_id = uav_id
        self.current_pos = np.array(initial_pos_xy[:2], dtype=float)  # 二维当前位置

        # 1. 将初始位置设为锚点
        self.anchor_point = np.array(initial_pos_xy[:2], dtype=float)
        self.activity_radius = 150.0  # 无人机围绕初始点活动的最大半径 (米) - 可调整
        # 可以考虑从config.py传入此参数

        # 移除之前的 self.waypoint, self.orbit_radius, self.approach_threshold

        if initial_vel_vector_xy and len(initial_vel_vector_xy) == 2:
            self.current_speed = np.linalg.norm(initial_vel_vector_xy)
            if self.current_speed > 1e-6:
                self.current_heading_xy = math.atan2(initial_vel_vector_xy[1], initial_vel_vector_xy[0])
            else:
                self.current_heading_xy = np.random.uniform(0, 2 * np.pi)
        else:
            self.current_speed = np.random.uniform(UAV_MIN_SPEED, UAV_MAX_SPEED)
            self.current_heading_xy = np.random.uniform(0, 2 * np.pi)

        self.mean_speed = GM_MEAN_SPEED
        # 初始目标朝向可以随机，或等于当前朝向
        self.target_heading_xy = self.current_heading_xy

    def update(self):
        # --- 1. 计算当前的目标朝向 (基于锚点和活动半径) ---
        vector_to_anchor = self.anchor_point - self.current_pos
        distance_to_anchor = np.linalg.norm(vector_to_anchor)

        if distance_to_anchor > self.activity_radius:
            # 如果超出了活动半径，强制目标朝向指向锚点 (初始位置)
            self.target_heading_xy = math.atan2(vector_to_anchor[1], vector_to_anchor[0])
        else:
            # 在活动半径内，允许目标朝向更自由地随机变化，以模拟局部探索
            # 例如，让目标朝向缓慢随机漂移
            # 可以设定一个最大角度变化率，比如每秒变化 M 度
            max_target_heading_change_per_sec = math.radians(30)  # 例如每秒最多变30度
            random_drift = np.random.uniform(-max_target_heading_change_per_sec * DELTA_T,
                                             max_target_heading_change_per_sec * DELTA_T)
            self.target_heading_xy = (self.target_heading_xy + random_drift) % (2 * np.pi)

        # --- 2. 高斯-马尔可夫过程更新实际速度和当前水平朝向 ---
        self.current_speed = (GM_ALPHA_SPEED * self.current_speed +
                              (1 - GM_ALPHA_SPEED) * self.mean_speed +
                              np.sqrt(1 - GM_ALPHA_SPEED ** 2) * np.random.normal(0, GM_STD_SPEED))
        self.current_speed = np.clip(self.current_speed, UAV_MIN_SPEED, UAV_MAX_SPEED)

        angle_diff = (self.target_heading_xy - self.current_heading_xy + math.pi) % (2 * math.pi) - math.pi
        max_mean_angle_change_per_step = math.radians(60) * DELTA_T  # 每步允许的朝向调整幅度
        effective_mean_heading_xy = self.current_heading_xy + np.clip(angle_diff, -max_mean_angle_change_per_step,
                                                                      max_mean_angle_change_per_step)

        rand_angle_change_xy = np.random.normal(0, math.radians(GM_STD_ANGLE_CHANGE))
        self.current_heading_xy = (GM_ALPHA_ANGLE * self.current_heading_xy +
                                   (1 - GM_ALPHA_ANGLE) * effective_mean_heading_xy +
                                   np.sqrt(1 - GM_ALPHA_ANGLE ** 2) * rand_angle_change_xy)
        self.current_heading_xy = self.current_heading_xy % (2 * np.pi)

        # --- 3. 计算基于当前速度和朝向的潜在下一步位置 ---
        vx = self.current_speed * math.cos(self.current_heading_xy)
        vy = self.current_speed * math.sin(self.current_heading_xy)

        potential_next_pos_x = self.current_pos[0] + vx * DELTA_T
        potential_next_pos_y = self.current_pos[1] + vy * DELTA_T

        # --- 4. 边界处理和最终位置更新 ---
        reflected = False

        if potential_next_pos_x < 0:
            self.current_pos[0] = 0;
            self.current_heading_xy = math.pi - self.current_heading_xy;
            reflected = True
        elif potential_next_pos_x > AREA_WIDTH:
            self.current_pos[0] = AREA_WIDTH;
            self.current_heading_xy = math.pi - self.current_heading_xy;
            reflected = True
        else:
            self.current_pos[0] = potential_next_pos_x

        if potential_next_pos_y < 0:
            self.current_pos[1] = 0;
            self.current_heading_xy = -self.current_heading_xy;
            reflected = True
        elif potential_next_pos_y > AREA_HEIGHT:
            self.current_pos[1] = AREA_HEIGHT;
            self.current_heading_xy = -self.current_heading_xy;
            reflected = True
        else:
            self.current_pos[1] = potential_next_pos_y

        self.current_heading_xy = self.current_heading_xy % (2 * np.pi)

        if reflected:
            # 反射后，强制目标朝向指向锚点，帮助离开边界
            vector_to_anchor_after_reflection = self.anchor_point - self.current_pos
            if np.linalg.norm(vector_to_anchor_after_reflection) > 1.0:
                self.target_heading_xy = math.atan2(vector_to_anchor_after_reflection[1],
                                                    vector_to_anchor_after_reflection[0])
            else:  # 如果反射点恰好在其锚点，随机一个朝向
                self.target_heading_xy = (self.current_heading_xy + math.pi / 2 + np.random.uniform(-0.5, 0.5)) % (
                            2 * np.pi)
            self.current_heading_xy = (self.current_heading_xy + np.random.uniform(-0.1, 0.1)) % (2 * np.pi)
            self.current_speed = max(self.current_speed, UAV_MIN_SPEED * 1.1)  # 确保有速度离开

        final_vx = self.current_speed * math.cos(self.current_heading_xy)
        final_vy = self.current_speed * math.sin(self.current_heading_xy)
        final_velocity_vector = np.array([final_vx, final_vy])

        return self.current_pos.tolist(), final_velocity_vector.tolist()