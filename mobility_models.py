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
        self.current_pos = np.array(initial_pos_xy[:2], dtype=float) # 确保是二维

        # 为每个UAV分配一个固定的二维航路点
        self.waypoint = np.array([
            random.uniform(AREA_WIDTH * 0.1, AREA_WIDTH * 0.9),
            random.uniform(AREA_HEIGHT * 0.1, AREA_HEIGHT * 0.9)
        ], dtype=float) # 二维航路点

        self.orbit_radius = 50.0
        self.approach_threshold = self.orbit_radius * 1.5

        if initial_vel_vector_xy and len(initial_vel_vector_xy) == 2:
            self.current_speed = np.linalg.norm(initial_vel_vector_xy)
            if self.current_speed > 1e-6 : # 避免速度为0时atan2出错
                self.current_heading_xy = math.atan2(initial_vel_vector_xy[1], initial_vel_vector_xy[0])
            else:
                self.current_heading_xy = np.random.uniform(0, 2 * np.pi)
        else:
            self.current_speed = np.random.uniform(UAV_MIN_SPEED, UAV_MAX_SPEED)
            self.current_heading_xy = np.random.uniform(0, 2 * np.pi)

        self.mean_speed = GM_MEAN_SPEED
        self.target_heading_xy = self.current_heading_xy

    def update(self):
        vector_to_waypoint_xy = self.waypoint - self.current_pos
        distance_to_waypoint_xy = np.linalg.norm(vector_to_waypoint_xy)

        if distance_to_waypoint_xy > 1e-6:
            if distance_to_waypoint_xy > self.approach_threshold:
                self.target_heading_xy = math.atan2(vector_to_waypoint_xy[1], vector_to_waypoint_xy[0])
            else:
                angle_to_wp = math.atan2(vector_to_waypoint_xy[1], vector_to_waypoint_xy[0])
                self.target_heading_xy = angle_to_wp + math.pi / 2 # 逆时针环绕
        else:
            self.target_heading_xy = (self.target_heading_xy + np.random.uniform(-0.2, 0.2)) % (2 * np.pi)


        self.current_speed = (GM_ALPHA_SPEED * self.current_speed +
                              (1 - GM_ALPHA_SPEED) * self.mean_speed +
                              np.sqrt(1 - GM_ALPHA_SPEED**2) * np.random.normal(0, GM_STD_SPEED))
        self.current_speed = np.clip(self.current_speed, UAV_MIN_SPEED, UAV_MAX_SPEED)

        angle_diff = (self.target_heading_xy - self.current_heading_xy + math.pi) % (2*math.pi) - math.pi
        max_mean_angle_change = math.radians(60) # 允许更快的平均朝向变化
        effective_mean_heading_xy = self.current_heading_xy + np.clip(angle_diff, -max_mean_angle_change, max_mean_angle_change)

        rand_angle_change_xy = np.random.normal(0, math.radians(GM_STD_ANGLE_CHANGE))
        self.current_heading_xy = (GM_ALPHA_ANGLE * self.current_heading_xy +
                                   (1 - GM_ALPHA_ANGLE) * effective_mean_heading_xy +
                                   np.sqrt(1 - GM_ALPHA_ANGLE**2) * rand_angle_change_xy)
        self.current_heading_xy = self.current_heading_xy % (2 * np.pi)

        vx = self.current_speed * math.cos(self.current_heading_xy)
        vy = self.current_speed * math.sin(self.current_heading_xy)
        velocity_vector = np.array([vx, vy]) # 二维速度向量

        self.current_pos[0] += vx * DELTA_T
        self.current_pos[1] += vy * DELTA_T

        if not (0 <= self.current_pos[0] <= AREA_WIDTH):
            self.current_pos[0] = np.clip(self.current_pos[0], 0, AREA_WIDTH)
            self.target_heading_xy = (self.target_heading_xy + math.pi + np.random.uniform(-0.1, 0.1)) % (2*math.pi) # 简单掉头并加点随机
        if not (0 <= self.current_pos[1] <= AREA_HEIGHT):
            self.current_pos[1] = np.clip(self.current_pos[1], 0, AREA_HEIGHT)
            self.target_heading_xy = (self.target_heading_xy + math.pi + np.random.uniform(-0.1, 0.1)) % (2*math.pi)

        return self.current_pos.tolist(), velocity_vector.tolist() # 返回二维列表