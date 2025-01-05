from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np

from mobile_env.core.entities import BaseStation, UserEquipment

EPSILON = 1e-16


class Channel:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    # 计算基站与用户设备之间的信号功率损耗
    def power_loss(self, bs: BaseStation, ue: UserEquipment) -> float:
        pass

    # 计算基站和用户设备之间的信噪比
    def calculateSNR(self, bs: BaseStation, ue: UserEquipment):
        loss = self.power_loss(bs, ue)
        power = 10 ** ((bs.tx_power - loss) / 10)
        return power / ue.noise

    # 计算基站信号覆盖范围的等值线
    def isoline(
        self,
        bs: BaseStation,    # 基站对象，代表这个基站是信号发射的源头
        ue_config: Dict,    # 用户设备(UserEquipment)的配置字典，用于初始化一个虚拟的用户设备
        map_bounds: Tuple,  # 表示地图的边界（宽度和高度），形如 (width, height)
        dthresh: float,     # 数据速率阈值，表示基站与用户设备之间传输数据的最低要求
        num: int = 32,      # 用于将 360 度分成多少份，以便计算信号的传播
    ):
        width, height = map_bounds

        # 创建一个虚拟用户设备用于计算不同位置的信号情况
        dummy = UserEquipment(None, **ue_config)

        isoline = []

        # 计算基站信号在不同方向上的最远覆盖范围，以绘制信号覆盖的等值线
        for theta in np.linspace(EPSILON, 2 * np.pi, num=num):
            # 地图边界的碰撞点坐标(x1, y1)
            x1, y1 = self.boundary_collison(theta, bs.x, bs.y, width, height)

            # 计算从基站位置(bs.x, bs.y)到信号与地图边界的碰撞点(x1, y1)之间的所有点的坐标
            slope = (y1 - bs.y) / (x1 - bs.x)
            xs = np.linspace(bs.x, x1, num=100)
            ys = slope * (xs - bs.x) + bs.y

            # 基站bs在给定位置point上的虚拟用户设备dummy所接收到的信号数据速率
            def drate(point):
                dummy.x, dummy.y = point
                snr = self.calculateSNR(bs, dummy)

                return self.datarate(bs, dummy, snr)

            points = zip(xs.tolist(), ys.tolist())
            # map()是Python内置的函数，用于对可迭代对象中的每个元素应用某个函数
            # map(drate, points)会对这些坐标点逐一调用drate函数
            datarates = np.asarray(list(map(drate, points)))

            # np.where()返回满足条件datarates > dthresh的索引
            (idx,) = np.where(datarates > dthresh)
            # 找到最远的满足条件的点 确定信号在这个方向上可以传播的最远距离
            idx = np.max(idx)

            isoline.append((xs[idx], ys[idx]))

        xs, ys = zip(*isoline)
        return xs, ys

    # 计算每个位置上的数据速率，然后找到速率超过 dthresh 的点，构成等值线
    @classmethod
    def datarate(cls, bs: BaseStation, ue: UserEquipment, snr: float):
        if snr > ue.snr_threshold:
            return bs.bw * np.log2(1 + snr)

        return 0.0

    # 计算信号与地图边界的碰撞点，确定信号可以到达的最远距离
    @classmethod
    def boundary_collison(
        cls, theta: float, x0: float, y0: float, width: float, height: float
    ) -> Tuple:
        # 计算信号与右边界碰撞的点的坐标
        right_x1, right_y1 = width, np.tan(theta) * (width - x0) + y0
        # 计算信号与地图上边界的碰撞点
        upper_x1, upper_y1 = (-1) * np.tan(theta - 1 / 2 * np.pi) * (
            height - y0
        ) + x0, height
        # 计算信号与地图左边界的碰撞点
        left_x1, left_y1 = 0.0, np.tan(theta) * (0.0 - x0) + y0
        lower_x1, lower_y1 = np.tan(theta - 1 / 2 * np.pi) * (y0 - 0.0) + x0, 0.0

        if theta == 0.0:
            return width, y0

        elif 0.0 < theta < 1 / 2 * np.pi:
            x1 = np.min((right_x1, upper_x1, width))
            y1 = np.min((right_y1, upper_y1, height))
            return x1, y1

        elif theta == 1 / 2 * np.pi:
            return x0, height

        elif 1 / 2 * np.pi < theta < np.pi:
            x1 = np.max((left_x1, upper_x1, 0.0))
            y1 = np.min((left_y1, upper_y1, height))
            return x1, y1

        elif theta == np.pi:
            return 0.0, y0

        elif np.pi < theta < 3 / 2 * np.pi:
            return np.max((left_x1, lower_x1, 0.0)), np.max((left_y1, lower_y1, 0.0))

        elif theta == 3 / 2 * np.pi:
            return x0, 0.0

        else:
            x1 = np.min((right_x1, lower_x1, width))
            y1 = np.max((right_y1, lower_y1, 0.0))
            return x1, y1


# Okumura-Hata 模型是基于经验的路径损耗模型
class OkumuraHata(Channel):
    def power_loss(self, bs: BaseStation, ue: UserEquipment):
        distance = bs.point.distance(ue.point)

        ch = (
            0.8
            + (1.1 * np.log10(bs.frequency) - 0.7) * ue.height
            - 1.56 * np.log10(bs.frequency)
        )
        tmp_1 = (
            69.55 - ch + 26.16 * np.log10(bs.frequency) - 13.82 * np.log10(bs.height)
        )
        tmp_2 = 44.9 - 6.55 * np.log10(bs.height)

        return tmp_1 + tmp_2 * np.log10(distance + EPSILON)
