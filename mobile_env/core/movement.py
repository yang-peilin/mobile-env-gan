from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np

from mobile_env.core.entities import UserEquipment


class Movement:
    def __init__(
            self, width: float, height: float, seed: int, reset_rng_episode: str, **kwargs
    ):
        # 表示地图的宽度和高度，用户设备将在该范围内运动
        self.width, self.height = width, height
        self.reset_rng_episode = reset_rng_episode

        # 表示是否在每个仿真回合结束时重置随机数生成器
        self.seed = seed
        self.rng = None

    # 在每个仿真回合结束后调用，用于重置用户设备的运动状态
    def reset(self) -> None:
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        pass

    @abstractmethod
    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        pass


class RandomWaypointMovement(Movement):
    def __init__(self, **kwargs):
        # 传递给父类的构造函数
        super().__init__(**kwargs)

        # 使用固定的随机数种子初始化随机数生成器
        self.rng = np.random.default_rng(self.seed)

        # 存储每个用户设备当前的目标位置
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = None
        # 存储每个用户设备的初始位置，确保在仿真开始时为每个设备提供一个固定的位置
        self.initial: Dict[UserEquipment, Tuple[float, float]] = None

    def reset(self) -> None:
        # 在每次仿真重置时，重新初始化随机数生成器以保证随机数序列一致
        self.rng = np.random.default_rng(self.seed)

        super().reset()
        self.waypoints = {}
        self.initial = {}

    # 该方法用于移动一个用户设备（UE）到目标位置（waypoint），并返回新的位置
    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        # 如果用户设备ue当前没有目标位置（waypoint），则为它生成一个新的目标位置
        if ue not in self.waypoints:
            wx = self.rng.uniform(0, self.width)
            wy = self.rng.uniform(0, self.height)
            self.waypoints[ue] = (wx, wy)

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.waypoints[ue])

        if np.linalg.norm(position - waypoint) <= ue.velocity:
            waypoint = self.waypoints.pop(ue)
            return waypoint

        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)

        return tuple(position)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        if ue not in self.initial:
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            self.initial[ue] = (x, y)

        x, y = self.initial[ue]
        return x, y
