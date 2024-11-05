import random
import pandas as pd
from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment


# 生成器函数，随机生成基站位置
def generate_base_stations(env_config):
    # num_stations = random.randint(5, 10)
    num_stations = 6
    stations = []
    for bs_id in range(num_stations):
        x = random.uniform(0, 200)
        y = random.uniform(0, 200)
        stations.append(BaseStation(bs_id=bs_id, pos=(x, y), **env_config["bs"]))
    return stations


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

        # 使用生成器函数生成基站
        stations = generate_base_stations(env_config)

        # 创建用户设备
        users = [
            UserEquipment(ue_id=i, **env_config["ue"])
            for i in range(5)
        ]

        super().__init__(stations, users, config, render_mode)

    def reset(self):
        super().reset()
        # 使用生成器函数重新生成基站位置
        env_config = self.default_config()
        self.stations = generate_base_stations(env_config)
        self.stations = {bs.bs_id: bs for bs in self.stations}

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
        utility_df.to_csv(utility_filename, index=False)