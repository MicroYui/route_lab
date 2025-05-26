# -*- coding: utf-8 -*-
# utils.py
import numpy as np
import math

def calculate_distance(pos1, pos2):
    """计算二维欧几里得距离"""
    # 确保输入是类似数组的，并且至少有两个元素
    p1 = np.array(pos1[:2]) # 只取前两个分量，以防万一传入了三维的
    p2 = np.array(pos2[:2])
    return np.linalg.norm(p1 - p2)

def ensure_utf8(func):
    """装饰器，确保函数输出和注释是UTF-8 (主要用于print)"""
    def wrapper(*args, **kwargs):
        # 在实际应用中，确保环境和终端支持UTF-8更重要
        # Python 3 默认字符串是Unicode
        return func(*args, **kwargs)
    return wrapper

# 可以在这里添加更多数学或转换函数