from abc import abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
from mobile_env.core.entities import UserEquipment


class Movement:
    def __init__(self, width: float, height: float, seed: int, reset_rng_episode: str, **kwargs):
        self.width, self.height = width, height
        self.reset_rng_episode = reset_rng_episode
        self.seed = seed
        self.rng = None
        # print("初始化Movement, 其中的 seed: ", self.seed)

    # 在每个仿真回合结束后调用，用于重置用户设备的运动状态
    def reset(self) -> None:
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)
            # print("调用Movement的reset函数 重置self.rng: ", self.seed, " 这个的reset_rng_episode应该是True: ", self.reset_rng_episode)

    @abstractmethod
    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        pass

    @abstractmethod
    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        pass


class RandomWaypointMovement(Movement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.userMoveDirection: Dict[UserEquipment, Tuple[float, float]] = {}
        self.userPositionInitial: Dict[UserEquipment, Tuple[float, float]] = {}

    def reset(self) -> None:
        super().reset()
        self.userMoveDirection = {}
        self.userPositionInitial = {}

    # 该方法用于移动一个用户设备（UE）到目标位置（waypoint），并返回新的位置
    def move(self, ue: UserEquipment) -> tuple[int, int] | tuple[Any]:
        # print("调用 move，现在的seed是 ", self.seed)
        if ue not in self.userMoveDirection:
            wx = int(self.rng.uniform(0, self.width))
            wy = int(self.rng.uniform(0, self.height))
            self.userMoveDirection[ue] = (wx, wy)
        # print(f"用户设备 {ue.ue_id} 的目标位置: {self.userMoveDirection[ue]}")

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.userMoveDirection[ue])

        # 如果已到达目标位置，生成新的目标位置
        if np.linalg.norm(position - waypoint) <= ue.velocity:
            waypoint = self.userMoveDirection.pop(ue)
            return waypoint

        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)
        position = np.round(position).astype(int)

        return tuple(position)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        # print(f"initial_position 随机种子: {self.seed}, RNG 状态: {self.rng}")
        if ue not in self.userPositionInitial:
            x = int(self.rng.uniform(0, self.width))
            y = int(self.rng.uniform(0, self.height))
            self.userPositionInitial[ue] = (x, y)

        x, y = self.userPositionInitial[ue]
        return x, y
