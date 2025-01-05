import numpy as np


# 计算网络中所有基站与用户设备之间的连接总数
def number_connections(sim):
    if sim.bs2ue_connections is None:
        print("sim.connections is 0")
        return 0  # 如果没有连接，返回0
    return sum([len(con) for con in sim.bs2ue_connections.values()])


# 计算所有基站连接的用户设备的总数量，避免重复计算同一个用户设备被多个基站连接
def number_connected(sim):
    return len(set.union(set(), *sim.bs2ue_connections.values()))


# 计算所有连接的用户设备的平均数据速率
def mean_datarate(sim):
    if not sim.allUserDataRates:
        return 0.0
    return np.mean(list(sim.allUserDataRates.values()))


# 计算所有用户设备的平均效用值
def mean_utility(sim):
    if not sim.ue_utilities:
        return sim.utilityModel.lower
    return np.mean(list(sim.ue_utilities.values()))