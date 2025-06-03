# -*- coding: utf-8 -*-
# config.py

# 仿真时间参数
TOTAL_SIMULATION_TIME = 3600  # 救援总时间 (T), 例如: 1 小时 = 3600秒
DELTA_T = 1                   # 决策时间间隔 (Δt), 例如: 1 秒

# UAV 基本参数
N_UAVS = 10                   # 无人机总数 (N) - 可根据 DRLFR/QTAR 调整
UAV_MAX_QUEUE_LENGTH = 10000    # 无人机的最大数据队列长度 (Q_max) - 包的数量
UAV_MAX_ENERGY = 15.0        # 无人机的最大电池容量 (E_max) - 能量单位，例如mJ
UAV_MIN_ENERGY = 0.0         # 无人机的最低安全电量 (E_min) - 能量单位
UAV_MAX_TRANS_POWER = 0.2     # 无人机转发时的最大功率 (P_max_trans) 单位: 瓦特 (例如 DRLFR 中的 100mW)
UAV_MAX_SPEED = 20.0            # 最大速度 (米/秒), 可根据 DRLFR/QTAR 调整
UAV_MIN_SPEED = 5.0            # 最小速度 (米/秒)

# 通信模型参数
RHO_0 = 1e-4                  # 参考距离下的信道功率 (假设值, G_i*G_j*lambda^2 / (4pi)^2)
                                # 需要根据天线增益 G 和波长 lambda 具体设定或校准
NOISE_POWER = 0.8e-12           # 噪声功率 (sigma^2) 单位: 瓦特 (例如, -90 dBm)
BANDWIDTH = 20e6              # 带宽 (B) 单位: 赫兹 (例如 DRLFR 中的 20 MHz)
MAX_COMMUNICATION_RANGE = 250.0 # 最大通信距离 (d_max) 单位: 米

# 李雅普诺夫优化参数
LYAPUNOV_BETA = 1.0e15          # 权重参数 β
LYAPUNOV_GAMMA = 0.1          # 权重参数 γ
LYAPUNOV_EPSILON = 1e-6         # 小的正数常量 ε (用于避免除零)
LYAPUNOV_V = 1.0e6             # 漂移加惩罚中的权衡参数 V

# Hello 消息参数
TAU_HELLO = 0.5               # Hello 间隔经验值 τ

# 移动模型参数 (高斯-马尔可夫)
GM_ALPHA_SPEED = 0.7          # 高斯-马尔可夫 速度平滑因子
GM_ALPHA_ANGLE = 0.7          # 高斯-马尔可夫 角度平滑因子
GM_MEAN_SPEED = (UAV_MAX_SPEED + UAV_MIN_SPEED) / 2 # 平均速度
GM_STD_SPEED = (UAV_MAX_SPEED - UAV_MIN_SPEED) / 6 # 速度标准差 (调小以避免极端值)
GM_STD_ANGLE_CHANGE = 15      # 角度变化标准差 (单位: 度)

# 区域范围 (用于UAV部署和移动)
AREA_WIDTH = 1000             # 区域宽度 (米)
AREA_HEIGHT = 1000            # 区域高度 (米)

USE_MANUAL_UAV_POSITIONS = True # True: 使用下面的列表; False: 随机生成初始位置
UAV_INITIAL_POSITIONS = [
    (200.0, 200.0), # UAV 0
    (200.0, 400.0), # UAV 1
    (400.0, 200.0), # UAV 2
    (400.0, 400.0), # UAV 3
    (400.0, 600.0), # UAV 4
    (400.0, 800.0), # UAV 5
    (600.0, 200.0), # UAV 6
    (600.0, 400.0), # UAV 7
    (600.0, 600.0), # UAV 8
    (600.0, 800.0)  # UAV 9
]

# 列出作为数据源的无人机的ID（从0开始计数）
DATA_SOURCE_UAV_IDS = [1, 5]

# 数据中心
# DATA_CENTER_POSITION = (AREA_WIDTH / 2, AREA_HEIGHT / 2) # 数据中心位置 (D)
DATA_CENTER_POSITION = (800, 500) # 数据中心位置 (D)

# 数据产生速率 (lambda_i(t)) - 每个 DELTA_T 产生的数据包数
DATA_GENERATION_RATE_AVG = 15  # 平均每个时间单位产生个数据包 (泊松分布均值)
DATA_PACKET_SIZE_BITS = 1024 * 8 # 每个数据包的大小 (例如, 1KB)

# 性能评价指标相关
ENERGY_BALANCE_WINDOW = 100 # 计算能量均衡度的移动窗口或最终计算

# 仿真控制
DEBUG_LEVEL = 1 # 0: 无, 1: 信息, 2: 详细

# 随机数种子
SIMULATION_SEED = 1