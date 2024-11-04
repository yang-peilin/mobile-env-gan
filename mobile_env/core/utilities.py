from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np


# 用于模拟移动通信网络中的用户体验质量 (QoE)
class Utility:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    # 用于根据用户设备的数据速率来计算效用值（QoE）
    @abstractmethod
    def utility(self, datarate) -> float:
        pass

    # 用于对效用值进行归一化
    @abstractmethod
    def scale(self, utility) -> float:
        pass

    # 用于对归一化后的效用值进行复原
    @abstractmethod
    def unscale(self, utility) -> float:
        pass


# 用于计算基于对数函数的效用值，并在给定的范围（lower 和 upper）内进行裁剪
class BoundedLogUtility(Utility):
    def __init__(
            self,
            lower: float,
            upper: float,                           # lower 和 upper：效用值的上下限，用于限制效用值的范围
            coeffs: Tuple[float, float, float],     # coeffs：对数函数的系数，用于计算效用值
            **kwargs: Dict
    ):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.coeffs = coeffs

    # 计算基于数据速率的效用值，其中 w1, w2, w3 是对数函数的系数
    def utility(self, datarate) -> float:
        w1, w2, w3 = self.coeffs
        if datarate <= 0.0:
            return self.lower

        # 使用np.clip()将效用值限制在self.lower和self.upper之间
        utility = np.clip(
            w1 * np.log(w2 + datarate) / np.log(w3), self.lower, self.upper
        )
        return utility

    # 缩放效用值
    def scale(self, utility) -> float:
        return 2 * (utility - self.lower) / (self.upper - self.lower) - 1

    # 反缩放效用值
    def unscale(self, utility) -> float:
        return (utility + 1) / 2 * (self.upper - self.lower) + self.lower
