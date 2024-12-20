import os
import string
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pygame import Surface

from mobile_env.core import metrics
from mobile_env.core.arrival import NoDeparture
from mobile_env.core.channels import OkumuraHata
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.logging import Monitor
from mobile_env.core.movement import RandomWaypointMovement
from mobile_env.core.schedules import ResourceFair
from mobile_env.core.util import BS_SYMBOL, deep_dict_merge
from mobile_env.core.utilities import BoundedLogUtility


class MComCore:
    NOOP_ACTION = 0
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, stations, users, config={}, render_mode=None):

        self.macro = None

        # 保存渲染模式
        self.render_mode = render_mode
        assert render_mode in self.metadata["render_modes"] + [None]

        # 递归地合并传入的配置字典 config 和默认配置字典
        config = deep_dict_merge(self.default_config(), config)
        # 初始化种子和随机数生成器（RNG）配置
        config = self.seeding(config)

        # 仿真环境的宽度和高度
        self.width, self.height = config["width"], config["height"]
        # 确保仿真中的随机事件（如用户移动）可以重现
        self.seed = config["seed"]
        # 决定是否在每个仿真回合结束时重置随机数生成器
        self.reset_rng_episode = config["reset_rng_episode"]

        # 用于定义用户设备的到达模式（例如NoDeparture类），决定用户设备何时请求服务
        self.arrival = config["arrival"](**config["arrival_params"])
        # 信道模型，决定基站与用户设备之间通信的特性（例如信号强度、干扰等）
        self.channel = config["channel"](**config["channel_params"])
        # 调度器，负责管理基站与用户设备之间的资源分配
        self.scheduler = config["scheduler"](**config["scheduler_params"])
        # 移动模式，定义用户设备在仿真环境中的移动方式
        self.movement = config["movement"](**config["movement_params"])
        # 效用函数，可能用于评估仿真过程中的某些性能指标（如用户满意度、数据吞吐量等）
        self.utility = config["utility"](**config["utility_params"])

        # define parameters that track the simulation's progress
        # 仿真的最大时间长度，定义了每轮仿真的持续时间
        self.EP_MAX_TIME = config["EP_MAX_TIME"]
        self.time = None
        self.closed = False

        # 将输入的基站列表转换为字典，键为基站的bs_id，值为基站对象
        self.stations = {bs.bs_id: bs for bs in stations}
        # 将用户设备列表转换为字典，键为用户设备的ue_id，值为用户对象
        self.users = {ue.ue_id: ue for ue in users}
        self.NUM_STATIONS = len(self.stations)
        self.NUM_USERS = len(self.users)

        # define sizes of base feature set that can or cannot be observed
        self.feature_sizes = {
            "connections": self.NUM_STATIONS,  # 表示每个基站的连接数或连接状态
            "snrs": self.NUM_STATIONS,  # 表示每个基站的 SNR 状态
            "utility": 1,
            "bcast": self.NUM_STATIONS,
            "stations_connected": self.NUM_STATIONS,  # 表示每个基站的连接状态
        }

        # 存储当前活跃的用户设备: 请求服务的用户设备，表示这些设备目前正在网络中使用资源
        self.active: List[UserEquipment] = None
        # 存储基站（BS）和用户设备（UE）之间的下行连接
        # self.bs2ue_connections = {
        #     bs1: {ue1, ue2, ue3},  # 基站 bs1 连接了用户设备 ue1、ue2 和 ue3
        #     bs2: {ue4},  # 基站 bs2 仅连接了用户设备 ue4
        #     bs3: set()  # 基站 bs3 没有连接任何用户设备
        # }
        self.bs2ue_connections: Dict[BaseStation, Set[UserEquipment]] = None
        # 存储基站和用户设备之间的下行数据速率
        # self.datarates = {
        #     (bs1, ue1): 12.5,  # 基站 bs1 到用户设备 ue1 的数据速率为 12.5 Mbps
        #     (bs1, ue2): 7.8,  # 基站 bs1 到用户设备 ue2 的数据速率为 7.8 Mbps
        #     (bs2, ue3): 5.2,  # 基站 bs2 到用户设备 ue3 的数据速率为 5.2 Mbps
        # }
        self.bs2ue_dataRates: Dict[Tuple[BaseStation, UserEquipment], float] = None
        # 存储每个用户设备的效用值（例如经过缩放的效用函数）
        self.utilities: Dict[UserEquipment, float] = None
        # 存储随机数生成器（RNG）
        self.rng = None

        # pygame中的窗口对象
        self.window = None
        # pygame中的时钟对象
        self.clock = None
        # 渲染基站与用户设备之间连接等值线的对象
        self.conn_isolines = None
        # 渲染移动设备相关的等值线的对象
        self.mb_isolines = None

        # 为仿真环境设置度量指标（metrics）
        # scalar_metrics：表示标量类型的度量，例如总连接数、用户设备数量等
        config["metrics"]["scalar_metrics"].update(
            {
                "number connections": metrics.number_connections,  # 基站和用户设备之间的总连接数
                "number connected": metrics.number_connected,  # 已连接的用户设备数量
                "mean utility": metrics.mean_utility,  # 用户设备的平均效用值
                "mean datarate": metrics.mean_datarate,  # 所有用户设备的平均数据速率
            }
        )
        # 在仿真过程中追踪各个度量指标的变化
        self.monitor = Monitor(**config["metrics"])

        # 初始化存储用户在所有时间步中的数据速率和效用值的字典
        self.all_step_datarates = {}
        self.all_step_utilities = {}

    @classmethod
    def default_config(cls):
        width, height = 200, 200
        ep_time = 20
        config = {
            # 环境的宽度和高度，默认设置为 200
            "width": width,
            "height": height,
            # 仿真回合的最大时间长度，默认值为100，表示每个仿真回合的持续时间
            "EP_MAX_TIME": ep_time,
            "seed": 2024,
            # 一个布尔值，表示是否在每个仿真回合结束时重置随机数生成器
            "reset_rng_episode": False,
            # 定义用户设备的到达和离开模式。NoDeparture 表示用户设备一旦加入网络，就不会离开
            "arrival": NoDeparture,
            # channel: OkumuraHata：信道模型，用于模拟基站和用户设备之间的信号传播损耗
            "channel": OkumuraHata,
            # scheduler: ResourceFair：调度器模型，用于管理基站与用户设备之间的资源分配
            "scheduler": ResourceFair,
            # movement: RandomWaypointMovement：用户设备的移动模型
            "movement": RandomWaypointMovement,
            # utility: BoundedLogUtility：效用函数模型，用于计算用户设备的体验质量（QoE）
            "utility": BoundedLogUtility,
            # bw：基站的带宽 单位为赫兹（Hz）
            # freq：基站的工作频率 单位为 MHz
            # tx：基站的发射功率，单位为dBm
            # height：基站的高度，单位为米
            "bs": {"bw": 9e6, "freq": 2500, "tx": 30, "height": 50},
            "ue": {
                # velocity：用户设备的移动速度，单位为米每秒
                "velocity": 1.5,
                # snr_tr：用户设备的SNR（信噪比）阈值
                "snr_tr": 2e-8,
                # 用户设备的噪声功率，表示信号接收时的背景噪声
                "noise": 1e-9,
                "height": 1.8,
            },
        }

        # 为仿真环境中不同的模块（如用户设备的到达模式、信道模型、调度器、移动模型和效用函数等）设置
        # 默认的参数，并将这些参数添加到仿真环境的配置字典config中
        # ep_time：表示仿真回合的最大时间长度
        aparams = {"ep_time": ep_time, "reset_rng_episode": False}
        config.update({"arrival_params": aparams})
        # 信道模型
        config.update({"channel_params": {}})
        # 调度器的参数 (scheduler_params)
        config.update({"scheduler_params": {}})
        # 移动模型的参数(movement_params)
        mparams = {
            "width": width,
            "height": height,
            "reset_rng_episode": False,
        }
        config.update({"movement_params": mparams})
        # 效用函数的参数(utility_params)
        uparams = {"lower": -20, "upper": 20, "coeffs": (10, 0, 10)}
        # coeffs：是一个元组，表示效用函数的系数
        config.update({"utility_params": uparams})

        # 为仿真环境的度量系统（metricssystem） 设置默认的配置参数
        config.update(
            {
                "metrics": {
                    # 标量度量指标: 用于存储 标量类型的度量指标 如 总连接数、总数据速率、平均效用值
                    "scalar_metrics": {},
                    # 用户设备度量指标: 用于存储 用户设备（UE）相关的度量指标 如 每个用户设备的 数据速率、信号强度、效用值
                    "ue_metrics": {},
                    # 基站度量指标: 用于存储 基站（BS）相关的度量指标 如 每个基站的 负载情况、连接的用户设备数量、总数据吞吐量
                    "bs_metrics": {},
                }
            }
        )

        return config

    @classmethod
    def seeding(cls, config):
        seed = config["seed"]
        keys = [
            "arrival_params",  # 用户设备的到达模式参数
            "channel_params",  # 信道模型参数
            "scheduler_params",  # 调度器参数
            "movement_params",  # 用户设备的移动模型参数
            "utility_params",  # 效用函数参数
        ]
        for num, key in enumerate(keys):
            if key not in config:
                config[key] = {}
            # 为每个模块的种子参数设置一个不同的值
            config[key]["seed"] = seed + num + 1

        return config

    # 用于在仿真环境中重置环境到初始状态
    # 该方法通常在仿真中的每个新回合开始时调用，用于清除上一次仿真的状态并初始化新的仿真状态
    # 包括时间、随机数生成器、用户设备状态（如位置、连接状态）、基站和用户设备之间的连接等初始化操作
    # 返回的结果是新的观测值（obs）和额外信息（info）
    # def reset(self, *, seed=None, options=None):
    def reset(self, *, seed=None):

        # 将仿真的时间重置为 0.0
        self.time = 0.0

        # 如果传入了seed参数，调用self.seeding()方法为环境中的不同模块设置随机数生成器种子
        if seed is not None:
            self.seeding({"seed": seed})
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # 调用arrival、channel、scheduler、movement和utility对象的reset()方法，重置这些模块的内部状态
        self.arrival.reset()
        self.channel.reset()
        self.scheduler.reset()
        self.movement.reset()
        self.utility.reset()

        # 为每个用户设备UE生成新的到达时间(stime)和离开时间(extime)，模拟用户设备何时加入和离开网络
        for ue in self.users.values():
            ue.stime = self.arrival.arrival(ue)
            ue.extime = self.arrival.departure(ue)

        # 为每个用户设备生成新的初始位置
        for ue in self.users.values():
            ue.x, ue.y = self.movement.initial_position(ue)

        self.active = [ue for ue in self.users.values() if ue.stime <= 0]
        self.active = sorted(self.active, key=lambda ue: ue.ue_id)

        # self.connections = {
        #     bs1: {ue1, ue2},  # 基站 bs1 连接了用户设备 ue1 和 ue2
        #     bs2: {ue3},  # 基站 bs2 仅连接了用户设备 ue3
        #     bs3: set()  # 基站 bs3 没有连接任何用户设备
        # }
        self.bs2ue_connections = defaultdict(set)

        # self.datarates = {
        #     (bs1, ue1): 10.5,  # 基站 bs1 向用户设备 ue1 分配的数据速率为 10.5 Mbps
        #     (bs1, ue2): 7.8,  # 基站 bs1 向用户设备 ue2 分配的数据速率为 7.8 Mbps
        #     (bs2, ue3): 5.2,  # 基站 bs2 向用户设备 ue3 分配的数据速率为 5.2 Mbps
        # }
        self.bs2ue_dataRates = defaultdict(float)

        # self.utilities = {
        #     ue1: 0.85,  # 用户设备 ue1 的效用值
        #     ue2: 0.75,  # 用户设备 ue2 的效用值
        #     ue3: -0.2,  # 用户设备 ue3 的效用值 (可能未连接或连接速率较低)
        # }
        self.utilities = {}

        self.max_departure = max(ue.extime for ue in self.users.values())

        # 重置监控器：调用monitor对象的reset()方法，重置上一轮仿真中的监控结果
        self.monitor.reset()

        # 清空数据，准备下一轮循环
        self.all_step_datarates = {}
        self.all_step_utilities = {}

    # 如果SNR超过了用户设备的最小门限，则连接可以建立
    def check_connectivity(self, bs: BaseStation, ue: UserEquipment) -> bool:
        snr = self.channel.snr(bs, ue)
        return snr > ue.snr_threshold

    # 只保留符合check_connectivity的连接
    def update_connections(self) -> None:
        connections = {
            bs: set(ue for ue in ues if self.check_connectivity(bs, ue))
            for bs, ues in self.bs2ue_connections.items()
        }
        self.bs2ue_connections.clear()
        self.bs2ue_connections.update(connections)

    # 用于在仿真环境中执行一个时间步的操作
    # 一个完整的时间步长中，用户设备与基站之间的交互、资源分配、效用计算和奖励生成的过程
    def step(self):

        # 更新基站与用户设备之间的连接状态，移除那些已经不再满足连接条件的连接
        self.update_connections()

        # 根据基站和用户设备的连接情况 重新分配每个基站与连接的用户设备之间的下行数据速率
        self.bs2ue_dataRates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.bs2ue_dataRates.update(drates)

        # 对用户设备的多个连接进行数据速率的聚合，计算每个用户设备的总数据速率
        self.macro = self.macro_datarates(self.bs2ue_dataRates)

        # 根据每个活跃用户设备的总数据速率，计算它们的效用值
        self.utilities = {
            ue: self.utility.utility(self.macro.get(ue, 0.0))  # 默认值 0.0，避免 KeyError
            for ue in self.active
        }

        # 将效用值缩放到[-1, 1]的范围
        self.utilities = {
            ue: self.utility.scale(util) for ue, util in self.utilities.items()
        }

        # 记录当前的用户指标
        self.monitor.update(self)

        # 记录当前时间步的特征数据
        obs = self.features()

        # 将当前时间步的特征数据添加到保存的列表中
        self.record_step_data(obs)

        # 调用移动模型 movement，根据用户设备的移动模式更新它们的位置
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # 找到符合 SNR 阈值的最近基站，并连接到该基站
        for ue in self.active:
            available_bs = [bs for bs in self.stations.values() if self.check_connectivity(bs, ue)]
            if available_bs:
                closest_bs = min(available_bs, key=lambda bs: np.linalg.norm([ue.x - bs.point.x, ue.y - bs.point.y]))

                # 首先断开该用户与所有基站的连接，保证只连接最近的基站
                for bs in self.bs2ue_connections:
                    if ue in self.bs2ue_connections[bs]:
                        self.bs2ue_connections[bs].remove(ue)

                # 如果最近的基站还没有连接上该用户，则连接
                if closest_bs not in self.bs2ue_connections:
                    self.bs2ue_connections[closest_bs] = set()
                self.bs2ue_connections[closest_bs].add(ue)

        # 找出离开仿真环境的用户设备，将它们从基站的连接列表中移除
        leaving = set([ue for ue in self.active if ue.extime <= self.time])
        for bs, ues in self.bs2ue_connections.items():
            self.bs2ue_connections[bs] = ues - leaving

        self.active = sorted(
            [
                ue
                for ue in self.users.values()
                if ue.extime > self.time and ue.stime <= self.time
            ],
            key=lambda ue: ue.ue_id,
        )

        # 将仿真时间步 + 1，向前推进一个时间步
        self.time += 1

        # 检查仿真结束：如果仿真时间已经结束且有可视化窗口，则关闭环境
        if self.time_is_up and self.window:
            self.close()

        if not self.active and not self.time_is_up:
            return self.step({})

        info = self.monitor.info()

        return info

    def record_step_data(self, obs):
        """记录当前时间步的所有用户特征数据到全局数据集中"""
        for ue_id, ue_features in obs.items():
            datarate = ue_features["datarate"][0]
            utility = ue_features["utility"][0]

            # 初始化每个用户的列表
            if ue_id not in self.all_step_datarates or not isinstance(self.all_step_datarates[ue_id], list):
                self.all_step_datarates[ue_id] = []
            if ue_id not in self.all_step_utilities or not isinstance(self.all_step_utilities[ue_id], list):
                self.all_step_utilities[ue_id] = []

            # 追加当前时间步的数据速率和效用值到对应的用户列表中
            self.all_step_datarates[ue_id].append(datarate)
            self.all_step_utilities[ue_id].append(utility)

    # 在数据收集完成后调用此方法来查看数据的维度并保存为CSV
    def save_all_step_data(self, idx):
        # 检查数据是否为空
        if not self.all_step_datarates or not self.all_step_utilities:
            print("数据记录为空，无法保存。")
            return

        try:
            # 数据速率记录
            max_datarates_steps = max(len(datarates) for datarates in self.all_step_datarates.values())
            # print(f"all_step_datarates: {len(self.all_step_datarates)} rows, {max_datarates_steps} columns")

            datarate_records = {"UE ID": list(self.all_step_datarates.keys())}
            for step in range(max_datarates_steps):
                datarate_records[f"Step {step + 1}"] = [
                    datarates[step] if step < len(datarates) else None
                    for datarates in self.all_step_datarates.values()
                ]
            data_rate_df = pd.DataFrame(datarate_records)

            # 保存到指定文件夹
            data_rate_file = os.path.join(
                "/Users/yangpeilin/NUS CE/project/mobile-env-gan/mobile_env/collectData/UserDataRate",
                f"dataRates_{idx}.csv")
            data_rate_df.to_csv(data_rate_file, index=False)

            # 效用值记录
            max_utilities_steps = max(len(utilities) for utilities in self.all_step_utilities.values())
            # print(f"all_step_utilities: {len(self.all_step_utilities)} rows, {max_utilities_steps} columns")

            utility_records = {"UE ID": list(self.all_step_utilities.keys())}
            for step in range(max_utilities_steps):
                utility_records[f"Step {step + 1}"] = [
                    utilities[step] if step < len(utilities) else None
                    for utilities in self.all_step_utilities.values()
                ]
            utility_df = pd.DataFrame(utility_records)

            # 保存到指定文件夹
            utility_file = os.path.join("/Users/yangpeilin/NUS CE/project/mobile-env-gan/mobile_env/collectData/UserQoE",
                                        f"all_step_utilities_{idx}.csv")
            utility_df.to_csv(utility_file, index=False)
            # print(f"数据保存完成：{data_rate_file} 和 {utility_file}")

        except Exception as e:
            print(f"保存数据失败: {e}")

    # 判断当前时间是否已经达到了最小的结束时间
    @property
    def time_is_up(self):
        return self.time >= min(self.EP_MAX_TIME, self.max_departure)

    # 遍历基站和用户设备之间的连接，聚合每个用户设备的总数据速率
    def macro_datarates(self, bs2ue_dataRates):
        ue_datarates = Counter()
        connected_ues = set()

        for (bs, ue), datarate in bs2ue_dataRates.items():
            # 检查当前用户是否已经连接到另一个基站
            assert ue not in connected_ues, f"用户设备 {ue} 多次连接，当前基站：{bs}"
            ue_datarates.update({ue: datarate})
            connected_ues.add(ue)

        return ue_datarates

    # 计算每个用户设备最终能接收到的下行数据速率
    def station_allocation(self, bs) -> Dict:
        # 获取与基站 bs 建立连接的用户集合
        if bs in self.bs2ue_connections:
            conns = self.bs2ue_connections[bs]
        else:
            conns = set()  # 如果基站不在 connections 中，使用空集合

        # 计算每个用户设备的SNR和最大数据速率
        snrs = [self.channel.snr(bs, ue) for ue in conns]

        # 计算每个用户设备的最大数据速率
        # zip(snrs, conns) = > [(snrs[0], conns[0]), (snrs[1], conns[1]), (snrs[2], conns[2]), ...]
        # max_allocation = [
        #     self.channel.datarate(bs, ue1, 10.5),
        #     self.channel.datarate(bs, ue2, 8.2),
        #     self.channel.datarate(bs, ue3, 5.6)
        # ]
        max_allocation = [
            self.channel.datarate(bs, ue, snr) for snr, ue in zip(snrs, conns)
        ]

        # rates是一个列表，存储了每个用户设备最终分配到的下行数据速率
        rates = self.scheduler.share(bs, max_allocation)

        # 返回每个用户设备与基站的最终数据速率
        return {(bs, ue): rate for ue, rate in zip(conns, rates)}

    # 计算每个基站（BaseStation）的平均效用值，并返回一个字典，表示各个基站的效用情况
    # {
    #     BaseStation1: UtilityValue1,
    #     BaseStation2: UtilityValue2,
    #     ...
    # }
    def station_utilities(self) -> Dict[BaseStation, UserEquipment]:
        idle = self.utility.scale(self.utility.lower)

        # 计算每个基站的平均效用值
        util = {
            bs: sum(self.utilities[ue] for ue in self.bs2ue_connections[bs]) / len(self.bs2ue_connections[bs])
            if self.bs2ue_connections[bs]
            else idle
            for bs in self.stations.values()
        }

        return util

    # 为每个基站计算其等值线并将这些结果存储在一个字典中
    def bs_isolines(self, drate: float) -> Dict:
        # 初始化一个空字典，用于存储每个基站的等值线数据
        isolines = {}

        # ["ue"]提取用户设备的默认配置
        config = self.default_config()["ue"]

        # 计算每个基站的等值线
        for bs in self.stations.values():
            isolines[bs] = self.channel.isoline(
                bs, config, (self.width, self.height), drate
            )

        return isolines

    # 生成用户设备（UE）在仿真环境中的特征观测值 并返回一个包含所有用户设备的特征字典
    # 包括连接状态、信噪比（SNR）、效用值（utility）、基站广播信息和连接用户设备的数量
    def features(self) -> Dict[int, Dict[str, np.ndarray]]:
        stations = sorted([bs for bs in self.stations.values()], key=lambda bs: bs.bs_id)

        # 调用station_utilities()方法，计算并返回每个基站的平均效用值
        bs_utilities = self.station_utilities()

        # 这是一个内部函数，用于为一个用户设备ue生成其观测特征
        def ue_features(ue: UserEquipment):
            connections = [bs for bs in stations if ue in self.bs2ue_connections[bs]]

            # 当前情况每个用户只可能连接最近的基站 所以加上这个assert
            assert len(connections) <= 1

            onehot = np.zeros(self.NUM_STATIONS, dtype=np.float32)
            onehot[[bs.bs_id for bs in connections]] = 1

            # 计算用户设备与每个基站之间的信噪比(SNR)，并对这些SNR值进行归一化处理，使其值在[0, 1]范围内
            snrs = [self.channel.snr(bs, ue) for bs in stations]
            maxsnr = max(snrs)
            snrs = np.asarray([snr / maxsnr for snr in snrs], dtype=np.float32)

            # 获取用户设备 ue 的效用值 (Utility)，并将该值转换为一个 NumPy 数组
            # 如果该用户设备的效用值未定义，则使用缩放后的默认效用值
            utility = (
                self.utilities[ue]
                if ue in self.utilities
                else self.utility.scale(self.utility.lower)
            )
            utility = np.asarray([utility], dtype=np.float32)

            # 这段代码的主要功能是构建用户设备与所有基站之间的广播效用值(Utility Broadcast)，
            # 用于描述用户设备在每个基站上的潜在效用值。如果用户设备未连接到某个基站，则该基站的效用值设置为默认效用值(idle)
            idle = self.utility.scale(self.utility.lower)
            util_bcast = {
                bs: util if self.check_connectivity(bs, ue) else idle
                for bs, util in bs_utilities.items()
            }
            util_bcast = np.asarray([util_bcast[bs] for bs in stations], dtype=np.float32)

            # 计算基站连接的用户设备数量
            def num_connected(bs):
                if self.check_connectivity(bs, ue):
                    return len(self.bs2ue_connections[bs])
                return 0.0

            # 遍历所有基站，计算每个基站连接的用户设备数量
            stations_connected = [num_connected(bs) for bs in stations]

            # 表示每个基站连接的用户设备数量相对于总连接数的比例
            total = max(1, sum(stations_connected))
            stations_connected = np.asarray(
                [num / total for num in stations_connected], dtype=np.float32
            )

            # 获取用户设备 ue 的数据速率 (Data Rate)，并将其 转换为 NumPy 数组
            datarate = np.asarray([self.macro.get(ue, 0.0)], dtype=np.float32)

            return {
                "connections": onehot,
                "snrs": snrs,
                "utility": utility,
                "bcast": util_bcast,
                "stations_connected": stations_connected,
                "datarate": datarate,  # 添加数据速率到特征中
            }

        # obs字典现在包含了所有用户设备的特征，无论它们是活跃还是非活跃
        obs = {ue.ue_id: ue_features(ue) for ue in self.users.values()}

        return obs

    # 计算基站等值线。
    # 设置 matplotlib 图表布局。
    # 渲染仿真内容（如基站、用户设备、数据速率等）。
    # 根据不同的渲染模式输出结果，或返回图像数组，或在窗口中显示图像。
    def render(self) -> None:
        mode = self.render_mode

        # 如果环境已经关闭，则停止渲染。
        if self.closed:
            return

        # 计算基站的连接范围等值线，即用户设备可以连接到基站的区域边界（基于信号强度）
        if self.conn_isolines is None:
            self.conn_isolines = self.bs_isolines(0.0)

        # 计算基站的1MB / s数据速率等值线，即用户设备可以接收到至少1MB / s数据速率的区域边界
        if self.mb_isolines is None:
            self.mb_isolines = self.bs_isolines(1.0)

        # 设置matplotlib图表的布局
        fig = plt.figure()
        fx = max(3.0 / 2.0 * 1.25 * self.width / fig.dpi, 8.0)
        fy = max(1.25 * self.height / fig.dpi, 5.0)
        plt.close()
        fig = plt.figure(figsize=(fx, fy))
        # 创建子图布局
        gs = fig.add_gridspec(
            ncols=2,  # 指定图表有两列
            nrows=3,  # 指定图表有三行
            width_ratios=(4, 2),  # 指定每列的宽度比例 (4, 2) 意味着第一列的宽度是第二列的两倍
            height_ratios=(2, 3, 3),  # 指定每行的高度比例 (2, 3, 3) 意味着第一行的高度较小，而第二行和第三行的高度相等
            hspace=0.45,  # 控制子图之间的垂直间距（行间距），以图形高度的比例表示
            wspace=0.2,  # wspace：控制子图之间的水平间距（列间距），以图形宽度的比例表示
            top=0.95,  # top=0.95：设置子图区域距离图形顶部的相对位置
            bottom=0.15,
            left=0.025,
            right=0.955,
        )

        # 用于显示仿真的主图，包括基站、用户设备的状态
        sim_ax = fig.add_subplot(gs[:, 0])
        # 用于显示仪表盘，可能包含一些整体的仿真数据
        dash_ax = fig.add_subplot(gs[0, 1])
        # 用于显示用户设备的体验质量（QoE）
        qoe_ax = fig.add_subplot(gs[1, 1])
        # 用于显示用户设备与基站的连接情况
        conn_ax = fig.add_subplot(gs[2, 1])

        # 只有当仿真时间大于 0 时，才进行渲染操作
        if self.time > 0:
            # 渲染仿真主图，显示基站、用户设备、连接等信息
            self.render_simulation(sim_ax)
            # 渲染仪表盘，可能显示一些仿真过程中的关键数据或指标
            self.render_dashboard(dash_ax)
            # 渲染用户设备的平均效用值（QoE），衡量用户体验
            self.render_mean_utility(qoe_ax)
            # 渲染用户设备与基站的连接状态图
            self.render_ues_connected(conn_ax)

        # 意味着对齐这两个子图的 y 轴标签
        fig.align_ylabels((qoe_ax, conn_ax))
        canvas = FigureCanvas(fig)
        canvas.draw()

        plt.close()

        # 将渲染后的图像转换为RGB格式的数组，可以用于视频录制或保存
        if mode == "rgb_array":
            # 将 matplotlib 画布中的 RGB 数据转换为一个 NumPy 数组
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            # 将之前获取的 RGB 数据数组重新调整形状，使其变为一个三维数组
            return data.reshape(canvas.get_width_height()[::-1] + (3,))

        # 通过pygame创建一个窗口来展示渲染结果
        elif mode == "human":
            # render RGBA image on pygame surface
            data = canvas.buffer_rgba()
            size = canvas.get_width_height()

            # set up pygame window to display matplotlib figure
            if self.window is None:
                pygame.init()
                self.clock = pygame.time.Clock()

                # set window size to figure's size in pixels
                window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
                self.window = pygame.display.set_mode(window_size)

                # remove pygame icon from window; set icon to empty surface
                pygame.display.set_icon(Surface((0, 0)))

                # set window's caption and background color
                pygame.display.set_caption("MComEnv")

            # clear surface
            self.window.fill("white")

            # plot matplotlib's RGBA frame on the pygame surface
            screen = pygame.display.get_surface()
            plot = pygame.image.frombuffer(data, size, "RGBA")
            screen.blit(plot, (0, 0))

            # update the full display surface to the window
            pygame.display.flip()

            # handle pygame events (such as closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        else:
            raise ValueError("Invalid rendering mode.")

    # 展示用户设备的效用值、基站的连接范围、数据速率等信息
    def render_simulation(self, ax) -> None:
        # colormap：使用matplotlib的颜色映射（RdYlGn），用于将效用值（QoE）映射为不同颜色
        colormap = cm.get_cmap("RdYlGn")
        # unorm：定义一个归一化对象，用于将效用值（self.utility.lower到self.utility.upper）映射为颜色
        unorm = plt.Normalize(self.utility.lower, self.utility.upper)

        # 绘制用户设备（UE）
        for ue, utility in self.utilities.items():
            utility = self.utility.unscale(utility)
            color = colormap(unorm(utility))

            ax.scatter(
                ue.point.x,
                ue.point.y,  # 分别表示用户设备的位置的 x 坐标和 y 坐标
                s=200,  # 设置散点的大小为 200
                zorder=2,
                color=color,
                marker="o",
            )
            # 在每个用户设备的旁边标注用户设备的ID
            ax.annotate(ue.ue_id, xy=(ue.point.x, ue.point.y), ha="center", va="center")

        # 遍历基站：遍历所有基站，并为每个基站绘制一个图标
        for bs in self.stations.values():
            ax.plot(
                bs.point.x,
                bs.point.y,
                marker=BS_SYMBOL,  # BS_SYMBOL：基站的图标
                markersize=30,  # markersize=30 表示基站图标的大小
                markeredgewidth=0.1,  # 设置标记边缘的宽度
                color="black",
            )
            # 将基站（BaseStation）的ID转换为对应的大写字母
            bs_id = string.ascii_uppercase[bs.bs_id]
            # ax.annotate()：在基站图标旁边显示基站的ID
            ax.annotate(
                bs_id,
                xy=(bs.point.x, bs.point.y),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )

            # 在图表上绘制基站连接范围的等值线
            # pdb.set_trace()
            # print("*self.conn_isolines[bs]: ", *self.conn_isolines[bs])
            # print(f"bs: {bs}, type(bs): {type(bs)}")

            ax.scatter(*self.conn_isolines[bs], color="gray", s=3)
            # 在图表上绘制基站的1MB / s数据速率等值线
            ax.scatter(*self.mb_isolines[bs], color="black", s=3)

        # 绘制基站和用户设备之间的连接
        for bs in self.stations.values():
            for ue in self.bs2ue_connections[bs]:

                # share = self.bs2ue_dataRates[(bs, ue)] / self.macro[ue]
                share = self.bs2ue_dataRates.get((bs, ue), 0.0) / self.macro.get(ue, 1.0)

                # 根据贡献比例，使用颜色映射为该连接分配颜色
                share = share * self.utility.unscale(self.utilities[ue])
                color = colormap(unorm(share))

                # ax.plot()：绘制基站和用户设备之间的连线，线条颜色表示该连接对用户设备效用的贡献
                ax.plot(
                    [ue.point.x, bs.point.x],
                    [ue.point.y, bs.point.y],
                    color=color,
                    # path_effects用于给绘制的线条添加一些特殊的视觉效果
                    path_effects=[
                        pe.SimpleLineShadow(shadow_color="black"),
                        pe.Normal(),
                    ],
                    # 设置线条的宽度
                    linewidth=3,
                    # 设置绘制元素的层次顺序
                    zorder=-1,
                )

        # 隐藏坐标轴与边界
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # 设置x轴和y轴的显示范围
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])

    # 用于在仿真过程中显示一个仪表盘，展示当前时间步和历史累计的平均数据速率和平均效用值
    def render_dashboard(self, ax) -> None:
        # mean_utilities: 从监控数据中提取包含仿真每个时间步的平均效用值的列表
        mean_utilities = self.monitor.scalar_results["mean utility"]
        # 当前时间步的平均效用值
        mean_utility = mean_utilities[-1]
        # 整个仿真过程中所有时间步的总体平均效用值
        total_mean_utility = np.mean(mean_utilities)

        # mean_datarates：从监控数据中提取包含仿真每个时间步的平均数据速率的列表
        # mean_datarate 当前时间步的平均数据速率
        # total_mean_datarate 整个仿真过程中所有时间步的平均数据速率
        mean_datarates = self.monitor.scalar_results["mean datarate"]
        mean_datarate = mean_datarates[-1]
        total_mean_datarate = np.mean(mean_datarates)

        # 隐藏轴和刻度
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 不显示 X 轴和 Y 轴的刻度和标签
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # "Current"：表示当前时间步的统计数据
        # "History"：表示历史累计的统计数据（整个仿真过程的平均值）
        rows = ["Current", "History"]
        # "Avg. DR [GB/s]"：表示平均数据速率
        # "Avg. Utility"：表示 平均效用值，衡量用户设备的体验质量
        cols = ["Avg. DR [GB/s]", "Avg. Utility"]
        # mean_datarate 和 mean_utility：表示当前时间步的平均数据速率和平均效用值
        # total_mean_datarate 和 total_mean_utility：表示整个仿真过程的历史平均数据速率和历史平均效用值
        text = [
            [f"{mean_datarate:.3f}", f"{mean_utility:.3f}"],
            [f"{total_mean_datarate:.3}", f"{total_mean_utility:.3f}"],
        ]

        # 创建并显示表格
        table = ax.table(
            text,  # 表格的内容，包含每个单元格的数据
            rowLabels=rows,  # 表格的行标签
            colLabels=cols,  # 表格的列标签
            cellLoc="center",  # 将表格单元格中的内容居中对齐
            edges="B",  # 只显示表格底部的边框，其他边框隐藏
            loc="upper center",  # 将表格放在图表的上方并居中
            bbox=[0.0, -0.25, 1.0, 1.25],  # 定义表格的位置和大小，使用 [x, y, width, height] 格式
        )
        # 调整表格的字体大小
        table.auto_set_font_size(False)
        table.set_fontsize(11)

    # 绘制仿真过程中平均效用值（Avg.Utility） 的变化曲线
    # 展示了在仿真过程中的每个时间步，所有用户设备的平均效用值随时间的变化趋势
    def render_mean_utility(self, ax) -> None:
        # time：生成一个时间步的数组，范围从0到当前的仿真时间self.time，用于作为X轴
        time = np.arange(self.time)
        # mean_utility：从监控对象 self.monitor 中获取记录的 平均效用值
        mean_utility = self.monitor.scalar_results["mean utility"]
        # ax.plot()：在指定的轴 ax 上绘制时间序列（X 轴）和对应的平均效用值（Y 轴）曲线
        ax.plot(time, mean_utility, linewidth=1, color="black")

        ax.set_ylabel("Avg. Utility")
        # 设置 X 轴和 Y 轴的范围
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([self.utility.lower, self.utility.upper])

    # 绘制仿真过程中 连接的用户设备数量 随时间变化的曲线
    def render_ues_connected(self, ax) -> None:
        time = np.arange(self.time)
        # ues_connected 表示在每个时间步上，有多少用户设备与基站保持连接
        ues_connected = self.monitor.scalar_results["number connected"]
        ax.plot(time, ues_connected, linewidth=1, color="black")

        ax.set_xlabel("Time")
        ax.set_ylabel("#Conn. UEs")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, len(self.users)])

    # 关闭仿真环境，同时终止其可视化的显示
    def close(self) -> None:
        pygame.quit()
        self.window = None
        self.closed = True
