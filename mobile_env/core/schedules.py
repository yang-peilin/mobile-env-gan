from abc import abstractmethod
from typing import List

from my_env_gan.core.entities import BaseStation


# 据速率资源分配给连接的用户设备
class Scheduler:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    # 用于定义如何将基站的可用速率资源分配给连接的用户设备
    @abstractmethod
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        pass


# ResourceFair 类 实现了一个资源公平调度策略
class ResourceFair(Scheduler):
    # share()方法：将基站的资源（即速率）平均分配给连接的用户设备
    # 返回每个用户设备分配到的速率，计算方法是将每个速率平分给所有连接的用户设备
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        return [rate / len(rates) for rate in rates]


# RateFair 类 实现了一种基于速率倒数加权的调度策略
class RateFair(Scheduler):
    # share()方法：根据连接的用户设备的速率，按速率倒数加权的方式分配资源。速率较低的用户设备会分配到更多的资源
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        total_inv_rate = sum([1 / rate for rate in rates])
        return 1 / total_inv_rate
