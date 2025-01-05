import json
import os
import random

import numpy as np
import pandas as pd
from sympy.integrals.benchmarks.bench_integrate import bench_integrate_sin

from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment

class MComCustom(MComCore):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config["ue"].update({
            "velocity": 10,
        })
        return config

    def __init__(self, config=None, render_mode=None):
        self.mb_iso_lines = None
        self.conn_iso_lines = None
        self.stationDict = None
        if config is None:
            config = {}
        self.default_config().update(config)

        # 基站位置每次回变化 所以这里设置一个空list
        stations = []

        # 创建用户设备
        users = [
            UserEquipment(ue_id=i, **self.default_config()["ue"])
            for i in range(7)
        ]

        super().__init__(stations, users, config, render_mode)

    def reset(self, *, seed=None):
        super().reset()

        # 设置随机种子
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # 初始化 基站stations
        stations = self.generate_base_stations(self.default_config())
        self.stationDict = {bs.bs_id: bs for bs in stations}
        self.NUM_STATIONS = len(self.stationDict)

        users = [ue for ue in self.userDict.values() if ue.startTime <= 0]
        self.activeUsers = sorted(users, key=lambda ue: ue.ue_id)
        self.NUM_USERS = len(self.userDict)

        self.conn_iso_lines = None
        self.mb_iso_lines = None

        # 初始化数据存储变量
        self.users_dataRateList = {ue.ue_id: [] for ue in self.userDict.values()}
        self.users_trajectoryList = {ue.ue_id: [] for ue in self.userDict.values()}
        
        # print("Base Station Positions:", [(bs.point.x, bs.point.y) for bs in self.stationDict.values()])
        # print("User Initial Positions:", [(ue.x, ue.y) for ue in self.userDict.values()])

    # 生成器函数，随机生成基站位置
    @staticmethod
    def generate_base_stations(env_config):
        num_stations = random.randint(5, 10)
        stations = []
        for bs_id in range(num_stations):
            x = int(random.uniform(0, 200))
            y = int(random.uniform(0, 200))
            station = BaseStation(bs_id=bs_id, pos=(x, y), **env_config["bs"])
            stations.append(station)
        return stations

    def save_base_station_positions(self, epoch_number):
        base_station_file = f"../collectData2/BaseStationPosition/stations_{epoch_number}.json"
        os.makedirs(os.path.dirname(base_station_file), exist_ok=True)

        positions = {bs.bs_id: (bs.point.x, bs.point.y) for bs in self.stationDict.values()}
        with open(base_station_file, "w") as f:
            json.dump(positions, f)
        # print(f"基站位置保存成功：{base_station_file}")