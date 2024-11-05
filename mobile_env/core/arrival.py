from abc import abstractmethod

import numpy as np

from mobile_env.core.entities import UserEquipment


class Arrival:
    def __init__(self, ep_time: int, seed: int, reset_rng_episode: bool, **kwargs):
        # 表示每个仿真周期的时间长度
        self.ep_time = ep_time
        self.seed = seed
        # 一个布尔值，控制是否在每个仿真周期重置随机数生成器
        self.reset_rng_episode = reset_rng_episode
        # 随机数生成器对象，初始值为 None，会在 reset 方法中进行初始化
        self.rng = None

    # 重置随机数生成器，用于生成用户设备到达和离开的随机时间
    def reset(self) -> None:
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    # 该方法的目的是计算或生成某个用户设备(ue)的到达时间
    def arrival(self, ue: UserEquipment) -> int:
        pass

    @abstractmethod
    # 该方法的目的是计算或生成某个用户设备的离开时间
    def departure(self, ue: UserEquipment) -> int:
        pass


class NoDeparture(Arrival):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def arrival(self, ue: UserEquipment) -> int:
        return 0

    def departure(self, ue: UserEquipment) -> int:
        return self.ep_time
