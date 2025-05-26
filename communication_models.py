# -*- coding: utf-8 -*-
# communication_models.py
import numpy as np
import math
from config import RHO_0, NOISE_POWER, BANDWIDTH
from utils import calculate_distance # calculate_distance 现在是二维的

def calculate_channel_gain(pos_i_xy, pos_j_xy): # 函数名和参数表明是二维
    """计算信道增益 h_ij = rho_0 / d_ij^2 (基于二维距离)"""
    d_ij = calculate_distance(pos_i_xy, pos_j_xy) # 使用二维距离
    if d_ij < 1e-3: # 避免除以零或距离过小导致增益过大
        # 可以返回一个非常大的数，或者根据实际情况设定一个上限
        # 实际上，两个无人机不应该在完全相同的位置
        return RHO_0 / (1e-3**2) # 使用一个最小距离的平方
    gain = RHO_0 / (d_ij**2)
    return gain

def calculate_sinr(P_trans_ij, h_ij, interference_power):
    """计算信噪比 SINR_ij = (h_ij * P_trans_ij) / (I_ij + sigma^2)"""
    # 此函数逻辑不变，因为它处理的是标量值
    denom = interference_power + NOISE_POWER
    if denom < 1e-18 : denom = 1e-18 # 避免除以零
    sinr = (h_ij * P_trans_ij) / denom
    return sinr

def calculate_transmission_rate(sinr_ij):
    """计算传输速率 r_ij = B * log2(1 + SINR_ij)"""
    # 此函数逻辑不变
    if sinr_ij <= 0:
        return 0.0
    rate = BANDWIDTH * math.log2(1 + sinr_ij)
    return rate # bits per second