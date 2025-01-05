from abc import abstractmethod
from typing import Dict, Tuple
import numpy as np


# 用于模拟移动通信网络中的用户体验质量 (QoE)
class Utility:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def calculateUtility(self, datarate) -> float:
        pass

    # 用于对效用值进行归一化
    @abstractmethod
    def scaleUtility(self, utility) -> float:
        pass

    # 用于对归一化后的效用值进行复原
    @abstractmethod
    def unscaleUtility(self, utility) -> float:
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

    # 计算基于数据速率(datarate)的效用值，并确保效用值在一定范围内
    def calculateUtility(self, datarate) -> float:
        w1, w2, w3 = self.coeffs
        if datarate <= 0.0:
            return self.lower

        utility = np.clip(
            w1 * np.log(w2 + datarate) / np.log(w3), self.lower, self.upper
        )
        return utility

    def scaleUtility(self, utility) -> float:
        return 2 * (utility - self.lower) / (self.upper - self.lower) - 1

    def unscaleUtility(self, utility) -> float:
        return (utility + 1) / 2 * (self.upper - self.lower) + self.lower
