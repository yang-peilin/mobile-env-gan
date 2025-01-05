from abc import abstractmethod

import numpy as np

from mobile_env.core.entities import UserEquipment


class Arrival:
    def __init__(self, ep_time: int, seed: int, reset_rng_episode: bool, **kwargs):
        self.ep_time = ep_time
        self.seed = seed
        self.reset_rng_episode = reset_rng_episode
        self.rng = None

    def reset(self) -> None:
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def setArrivalTime(self, ue: UserEquipment) -> int:
        pass

    @abstractmethod
    def setDepartureTime(self, ue: UserEquipment) -> int:
        pass


class NoDeparture(Arrival):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setArrivalTime(self, ue: UserEquipment) -> int:
        return 0

    def setDepartureTime(self, ue: UserEquipment) -> int:
        return self.ep_time
