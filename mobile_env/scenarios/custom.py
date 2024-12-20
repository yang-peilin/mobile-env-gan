import json
import os
import random
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

    def __init__(self, config={}, render_mode=None):
        env_config = self.default_config()
        env_config.update(config)

        # 基站位置每次回变化 所以这里设置一个空list
        stations = []

        # 创建用户设备
        users = [
            UserEquipment(ue_id=i, **env_config["ue"])
            for i in range(5)
        ]

        super().__init__(stations, users, config, render_mode)

    def reset(self, *, seed=None):
        super().reset()
        env_config = self.default_config()
        stations = self.generate_base_stations(env_config)
        self.stations = {bs.bs_id: bs for bs in stations}
        self.NUM_STATIONS = len(self.stations)
        self.NUM_USERS = len(self.users)
        self.conn_isolines = None
        self.mb_isolines = None

    # 生成器函数，随机生成基站位置
    @staticmethod
    def generate_base_stations(env_config):
        num_stations = random.randint(5, 10)
        stations = []
        for bs_id in range(num_stations):
            x = random.uniform(0, 200)
            y = random.uniform(0, 200)
            station = BaseStation(bs_id=bs_id, pos=(x, y), **env_config["bs"])
            stations.append(station)
        return stations

    def save_stations_to_file(self, iteration):
        stations_list = list(self.stations.values())
        stations_info = [
            {
                "bs_id": bs.bs_id,
                "x": bs.x,
                "y": bs.y
            } for bs in stations_list
        ]

        # 定义保存路径
        save_path = "/Users/yangpeilin/NUS CE/project/mobile-env-gan/mobile_env/collectData/baseStationPosition"
        filename = os.path.join(save_path, f'stations_info_{iteration}.json')

        # 保存到指定文件夹
        with open(filename, 'w') as f:
            json.dump(stations_info, f, indent=4)

    def save_results_to_file(self, datarate_filename="datarates_results.csv", utility_filename="utilities_results.csv"):

        # 确定所有用户的最大步数
        max_steps = max(len(datarates) for datarates in self.all_step_datarates.values())

        # 处理数据速率结果
        datarate_records = {"UE ID": list(self.all_step_datarates.keys())}
        for step in range(max_steps):
            datarate_records[f"Step {step + 1}"] = [
                datarates[step] if step < len(datarates) else None
                for datarates in self.all_step_datarates.values()
            ]

        # 转换为 DataFrame 并保存到 CSV 文件中
        datarate_df = pd.DataFrame(datarate_records)
        # 打印数据速率结果的行数和列数
        print(f"Datarate DataFrame: {datarate_df.shape[0]} rows, {datarate_df.shape[1]} columns")
        datarate_df.to_csv(datarate_filename, index=False)

        # 处理效用值结果
        utility_records = {"UE ID": list(self.all_step_utilities.keys())}
        for step in range(max_steps):
            utility_records[f"Step {step + 1}"] = [
                utilities[step] if step < len(utilities) else None
                for utilities in self.all_step_utilities.values()
            ]

        # 转换为 DataFrame 并保存到 CSV 文件中
        utility_df = pd.DataFrame(utility_records)
        # 打印效用值结果的行数和列数
        print(f"Utility DataFrame: {utility_df.shape[0]} rows, {utility_df.shape[1]} columns")
        utility_df.to_csv(utility_filename, index=False)
