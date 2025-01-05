from abc import abstractmethod
from typing import List
from mobile_env.core.entities import BaseStation


# 据速率资源分配给连接的用户设备
class Scheduler:
    def __init__(self, **kwargs):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        pass


# ResourceFair 类 实现了一个资源公平调度策略 默认用的是这个
class ResourceFair(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        return [rate / len(rates) for rate in rates]


# RateFair 类 实现了一种基于速率倒数加权的调度策略
class RateFair(Scheduler):
    def share(self, bs: BaseStation, rates: List[float]) -> List[float]:
        total_inv_rate = sum([1 / rate for rate in rates])
        return 1 / total_inv_rate
