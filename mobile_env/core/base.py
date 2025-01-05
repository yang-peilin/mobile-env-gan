import json
import os
import string
from collections import Counter, defaultdict
from traceback import TracebackException
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

    def __init__(
            self,
            stations: list[BaseStation],
            users: list[UserEquipment],
            config=None,
            render_mode=None
    ):

        self.max_departure = None
        if config is None:
            config = {}
        self.render_mode = render_mode
        assert render_mode in self.metadata["render_modes"] + [None]

        # 初始化config
        config = deep_dict_merge(self.default_config(), config)
        config = self.seeding(config)

        # 定义宽高和随机种子
        self.width, self.height = config["width"], config["height"]
        self.seed = config["seed"]
        self.reset_rng_episode = config["reset_rng_episode"]
        self.rng = None

        # 定义使用模型
        self.arrivalModel = config["arrival"](**config["arrival_params"])
        self.channelModel = config["channel"](**config["channel_params"])
        self.schedulerModel = config["scheduler"](**config["scheduler_params"])
        self.movementModel = config["movement"](**config["movement_params"])
        self.utilityModel = config["utility"](**config["utility_params"])

        # define parameters that track the simulation's progress
        self.EP_MAX_TIME = config["EP_MAX_TIME"]
        self.time = None
        self.closed = False

        # 初始化用户和基站
        self.stationDict = {bs.bs_id: bs for bs in stations}
        self.userDict = {ue.ue_id: ue for ue in users}
        self.NUM_STATIONS = len(self.stationDict)
        self.NUM_USERS = len(self.userDict)

        # 定义用户和基站之间的关系
        self.activeUsers: List[UserEquipment] = []
        self.bs2ue_connections: Dict[BaseStation, Set[UserEquipment]] = {}
        self.bs2ue_dataRates: Dict[Tuple[BaseStation, UserEquipment], float] = {}
        self.ue_utilities: Dict[UserEquipment, float] = {}
        self.allUserDataRates = None

        # pygame中的窗口对象
        self.window = None
        self.clock = None
        self.conn_iso_lines = None
        self.mb_iso_lines = None

        # 为仿真环境设置度量指标（metrics）
        config["metrics"]["scalar_metrics"].update(
            {
                "number connections": metrics.number_connections,  # 基站和用户设备之间的总连接数
                "number connected": metrics.number_connected,  # 已连接的用户设备数量
                "mean utility": metrics.mean_utility,  # 用户设备的平均效用值
                "mean datarate": metrics.mean_datarate,  # 所有用户设备的平均数据速率
            }
        )
        self.monitor = Monitor(**config["metrics"])

        self.users_trajectoryList = None
        self.users_dataRateList = None
        self.userQoEList = None

    @classmethod
    def default_config(cls):
        width, height = 200, 200
        ep_time = 20
        config = {
            "width": width,
            "height": height,
            "EP_MAX_TIME": ep_time,
            "seed": 2024,
            "reset_rng_episode": False,
            "arrival": NoDeparture,
            "channel": OkumuraHata,
            "scheduler": ResourceFair,
            "movement": RandomWaypointMovement,
            "utility": BoundedLogUtility,
            "bs": {"bw": 9e6, "freq": 2500, "tx": 40, "height": 50},
            "ue": {
                "velocity": 1.5,
                "snr_tr": 2e-8,
                "noise": 1e-9,
                "height": 1.6,
            },
        }

        aparams = {"ep_time": ep_time, "reset_rng_episode": False}
        config.update({"arrival_params": aparams})
        config.update({"channel_params": {}})
        config.update({"scheduler_params": {}})
        mparams = {
            "width": width,
            "height": height,
            "reset_rng_episode": True,
        }
        config.update({"movement_params": mparams})
        uparams = {"lower": -20, "upper": 20, "coeffs": (10, 0, 10)}
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
            config[key]["seed"] = seed + num + 1

        return config

    def reset(self, *, seed=None):

        # 将仿真的时间重置为 0.0
        self.time = 0.0

        # 如果传入了seed参数，调用self.seeding()方法为环境中的不同模块设置随机数生成器种子
        if seed is not None:
            self.seed = seed  # 保存固定种子
            # self.seeding({"seed": seed})

        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # 调用arrival、channel、scheduler、movement和utility对象的reset()方法，重置这些模块的内部状态
        self.arrivalModel.reset()
        self.channelModel.reset()
        self.schedulerModel.reset()
        self.movementModel.reset()
        self.utilityModel.reset()

        # 为每个用户设备UE生成新的到达时间(stime)和离开时间(extime)，模拟用户设备何时加入和离开网络
        for ue in self.userDict.values():
            ue.startTime = self.arrivalModel.setArrivalTime(ue)
            ue.exitTime = self.arrivalModel.setDepartureTime(ue)

        # 为每个用户设备生成新的初始位置
        for ue in self.userDict.values():
            ue.x, ue.y = self.movementModel.initial_position(ue)

        self.bs2ue_connections = defaultdict(set)
        self.bs2ue_dataRates = defaultdict(float)
        self.ue_utilities = {}
        self.max_departure = max(ue.exitTime for ue in self.userDict.values())

        # 重置监控器：调用monitor对象的reset()方法，重置上一轮仿真中的监控结果
        self.monitor.reset()

        self.userQoEList = {ue.ue_id: [] for ue in self.userDict.values()}  # 初始化每个用户的 QoE 列表

    # 如果SNR超过了用户设备的最小门限，则连接可以建立
    def check_connectivity(self, bs: BaseStation, ue: UserEquipment) -> bool:
        snr = self.channelModel.calculateSNR(bs, ue)
        return snr > ue.snr_threshold

    def available_connections(self, ue: UserEquipment) -> Set:
        stations = self.stationDict.values()
        return {bs for bs in stations if self.check_connectivity(bs, ue)}

    # 只保留符合check_connectivity的连接
    def update_connections(self) -> None:
        bs2ue_connections = {
            bs: set(ue for ue in ues if self.check_connectivity(bs, ue))
            for bs, ues in self.bs2ue_connections.items()
        }
        self.bs2ue_connections.clear()
        self.bs2ue_connections.update(bs2ue_connections)

    # 用于在仿真环境中执行一个时间步的操作
    def step(self, epoch_number, curr_step):
        # 调用移动模型，根据用户设备的移动模式更新位置
        for ue in self.activeUsers:
            ue.x, ue.y = self.movementModel.move(ue)

        # 找到符合 SNR 阈值的最近基站，并连接到该基站
        self.bs2ue_connections = defaultdict(set)  # 清空连接
        for ue in self.activeUsers:
            available_bs = [bs for bs in self.stationDict.values() if self.check_connectivity(bs, ue)]
            if available_bs:
                closest_bs = min(available_bs, key=lambda bs: np.linalg.norm([ue.x - bs.point.x, ue.y - bs.point.y]))
                self.bs2ue_connections[closest_bs].add(ue)

        # 计算数据速率
        self.bs2ue_dataRates = {}
        for bs in self.stationDict.values():
            drates = self.allocateDataRate2User(bs)
            self.bs2ue_dataRates.update(drates)

        # allUserDatarates 存储每个用户设备的总数据速率的字典
        self.allUserDataRates = self.user_total_datarates(self.bs2ue_dataRates)

        # 更新用户的效用值QoE
        self.ue_utilities = {
            ue: self.utilityModel.scaleUtility(
                self.utilityModel.calculateUtility(self.allUserDataRates.get(ue, 0.0))
            )
            for ue in self.activeUsers
        }

        # 保存当前基站布局、用户位置和用户-基站速率连接
        self.save_layout_and_data_rates(epoch_number, curr_step)

        # 更新每个用户的数据速率和运动轨迹
        for ue in self.activeUsers:
            datarate = self.allUserDataRates.get(ue, 0.0)
            self.users_dataRateList[ue.ue_id].append(round(datarate, 2))
            self.users_trajectoryList[ue.ue_id].append((ue.x, ue.y))
            qoe = self.ue_utilities.get(ue, 0.0)  # 获取用户的当前 QoE
            self.userQoEList[ue.ue_id].append(round(qoe, 2))  # 将 QoE 追加到对应用户的列表中

        # 记录当前的用户指标
        self.monitor.update(self)

        # # 打印调试信息
        # for ue in self.userDict.values():
        #     datarate = self.allUserDataRates.get(ue, 0.0)
        #     utility = self.ue_utilities.get(ue, self.utilityModel.scaleUtility(self.utilityModel.lower))
        #     print(f"UE ID: {ue.ue_id}, Data Rate: {datarate}, Utility: {utility}")

        self.time += 1

        # 移除离开仿真环境的用户
        leaving = set([ue for ue in self.activeUsers if ue.exitTime <= self.time])
        for bs, ues in self.bs2ue_connections.items():
            self.bs2ue_connections[bs] = ues - leaving

        # 更新活跃用户列表
        self.activeUsers = sorted(
            [ue for ue in self.userDict.values() if ue.exitTime > self.time >= ue.startTime],
            key=lambda ue: ue.ue_id,
        )

        if self.time_is_up and self.window:
            self.close()

        return

    def save_layout_and_data_rates(self, epoch_number, curr_step):
        # 保存基站位置
        base_station_positions = [
            {"bs_id": bs.bs_id, "x": round(float(bs.point.x), 2), "y": round(float(bs.point.y), 2)}
            for bs in self.stationDict.values()
        ]
        base_station_file = os.path.join(
            "..", "collectData", "BaseStationPosition", f"stations_info_{epoch_number}_{curr_step}.json"
        )
        os.makedirs(os.path.dirname(base_station_file), exist_ok=True)
        with open(base_station_file, "w") as f:
            json.dump(base_station_positions, f, indent=4)
        # print(f"基站布局已保存: {base_station_file}")

        # 保存用户位置
        user_positions = [
            {"ue_id": ue.ue_id, "x": round(float(ue.x), 2), "y": round(float(ue.y), 2)}
            for ue in self.userDict.values()
        ]
        user_position_file = os.path.join(
            "..", "collectData", "UserEquipmentPosition", f"user_positions_{epoch_number}_{curr_step}.json"
        )
        os.makedirs(os.path.dirname(user_position_file), exist_ok=True)
        with open(user_position_file, "w") as f:
            json.dump(user_positions, f, indent=4)
        # print(f"用户位置已保存: {user_position_file}")

        # 保存用户-基站数据速率
        user_bs_rates = [
            {"ue_id": ue.ue_id, "bs_id": bs.bs_id, "data_rate": round(float(rate), 2)}
            for (bs, ue), rate in self.bs2ue_dataRates.items()
        ]
        data_rate_file = os.path.join(
            "..", "collectData", "DataRate", f"data_rates_{epoch_number}_{curr_step}.json"
        )
        os.makedirs(os.path.dirname(data_rate_file), exist_ok=True)
        with open(data_rate_file, "w") as f:
            json.dump(user_bs_rates, f, indent=4)
        # print(f"用户-基站数据速率已保存: {data_rate_file}")

        # 保存用户效用值 QoE
        user_qoe = [
            {"ue_id": ue.ue_id, "qoe": round(self.ue_utilities.get(ue, 0.0), 2)}
            for ue in self.userDict.values()
        ]
        user_qoe_file = os.path.join(
            "..", "collectData", "UserQoE", f"user_qoe_{epoch_number}_{curr_step}.json"
        )
        os.makedirs(os.path.dirname(user_qoe_file), exist_ok=True)
        with open(user_qoe_file, "w") as f:
            json.dump(user_qoe, f, indent=4)
        # print(f"用户效用值已保存: {user_qoe_file}")

    def save_epoch_data(self, epoch_number):
        # 检查是否有用户数据速率
        if not self.users_dataRateList:
            print(f"警告：在第 {epoch_number} 轮中没有用户数据速率，跳过保存。")
            return

        # 保存数据速率
        datarate_file = os.path.join(
            "..", "collectData2", "DataRate", f"datarates_{epoch_number}.csv"
        )
        os.makedirs(os.path.dirname(datarate_file), exist_ok=True)
        datarates = [
            {"User ID": ue_id, "Data Rates": rates}
            for ue_id, rates in self.users_dataRateList.items()
        ]
        datarate_df = pd.DataFrame(datarates)
        datarate_df.to_csv(datarate_file, index=False)
        # print(f"数据速率保存成功：{datarate_file}")

        # 检查是否有用户运动轨迹
        if not self.users_trajectoryList:
            print(f"警告：在第 {epoch_number} 轮中没有用户运动轨迹，跳过保存。")
            return

        # 保存用户运动轨迹
        trajectory_file = os.path.join(
            "..", "collectData2", "UserEquipmentPosition", f"user_positions_{epoch_number}.csv"
        )
        os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)
        trajectories = [
            {"User ID": ue_id, "Trajectory": positions}
            for ue_id, positions in self.users_trajectoryList.items()
        ]
        trajectory_df = pd.DataFrame(trajectories)
        trajectory_df.to_csv(trajectory_file, index=False)
        # print(f"用户轨迹保存成功：{trajectory_file}")

        # 检查是否有用户效用值 QoE
        if not self.userQoEList:
            print(f"警告：在第 {epoch_number} 轮中没有用户效用值，跳过保存。")
            return

        # 保存用户效用值 QoE
        qoe_file = os.path.join(
            "..", "collectData2", "UserQoE", f"user_qoe_{epoch_number}.csv"
        )
        os.makedirs(os.path.dirname(qoe_file), exist_ok=True)
        qoes = [
            {"User ID": ue_id, "QoE": qoes}
            for ue_id, qoes in self.userQoEList.items()
        ]
        qoe_df = pd.DataFrame(qoes)
        qoe_df.to_csv(qoe_file, index=False)
        # print(f"用户效用值保存成功：{qoe_file}")

    # 判断当前时间是否已经达到了最小的结束时间
    @property
    def time_is_up(self):
        return self.time >= min(self.EP_MAX_TIME, self.max_departure)

    # 遍历基站和用户设备之间的连接，聚合每个用户设备的总数据速率
    # self.macro: 用来存储每个用户设备的总数据速率
    def user_total_datarates(self, bs2ue_dataRates):
        ue_dataRates = Counter()
        for (bs, ue), datarate in bs2ue_dataRates.items():
            ue_dataRates.update({ue: datarate})

        return ue_dataRates

    # 计算每个用户设备最终能接收到的下行数据速率
    def allocateDataRate2User(self, bs) -> Dict:
        if bs in self.bs2ue_connections:
            conns = self.bs2ue_connections[bs]
        else:
            conns = set()  # 如果基站不在 connections 中，使用空集合

        snrs = [self.channelModel.calculateSNR(bs, ue) for ue in conns]
        # max_allocation：一个列表，其中每个元素是基站与其连接用户之间的理论最大数据速率
        max_allocation = [
            self.channelModel.datarate(bs, ue, snr) for snr, ue in zip(snrs, conns)
        ]
        rates = self.schedulerModel.share(bs, max_allocation)

        # 对最终的速率进行格式化，保留两位小数
        return {(bs, ue): round(rate, 2) for ue, rate in zip(conns, rates)}

    # 计算每个基站（BaseStation）的平均效用值，并返回一个字典，表示各个基站的效用情况
    def allStationUtilities(self) -> Dict[BaseStation, UserEquipment]:
        idle = self.utilityModel.scaleUtility(self.utilityModel.lower)
        util = {
            bs: sum(self.ue_utilities[ue] for ue in self.bs2ue_connections[bs]) / len(self.bs2ue_connections[bs])
            if self.bs2ue_connections[bs]
            else idle
            for bs in self.stationDict.values()
        }

        return util

    # 为每个基站计算其等值线并将这些结果存储在一个字典中
    def bs_isolines(self, drate: float) -> Dict:
        isolines = {}
        config = self.default_config()["ue"]

        # 计算每个基站的等值线
        for bs in self.stationDict.values():
            isolines[bs] = self.channelModel.isoline(
                bs, config, (self.width, self.height), drate
            )

        return isolines

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
        if self.conn_iso_lines is None:
            self.conn_iso_lines = self.bs_isolines(0.0)

        # 计算基站的1MB / s数据速率等值线，即用户设备可以接收到至少1MB / s数据速率的区域边界
        if self.mb_iso_lines is None:
            self.mb_iso_lines = self.bs_isolines(1.0)

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

        # # 意味着对齐这两个子图的 y 轴标签
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
        unorm = plt.Normalize(self.utilityModel.lower, self.utilityModel.upper)

        # 绘制用户设备（UE）
        for ue, utility in self.ue_utilities.items():
            utility = self.utilityModel.unscaleUtility(utility)
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
        for bs in self.stationDict.values():
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

            ax.scatter(*self.conn_iso_lines[bs], color="gray", s=3)
            # 在图表上绘制基站的1MB / s数据速率等值线
            ax.scatter(*self.mb_iso_lines[bs], color="black", s=3)

        # 绘制基站和用户设备之间的连接
        for bs in self.stationDict.values():
            for ue in self.bs2ue_connections[bs]:

                # share = self.bs2ue_dataRates[(bs, ue)] / self.macro[ue]
                share = self.bs2ue_dataRates.get((bs, ue), 0.0) / self.allUserDataRates.get(ue, 1.0)

                # 根据贡献比例，使用颜色映射为该连接分配颜色
                share = share * self.utilityModel.unscaleUtility(self.ue_utilities[ue])
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
        ax.set_ylim([self.utilityModel.lower, self.utilityModel.upper])

    # 绘制仿真过程中 连接的用户设备数量 随时间变化的曲线
    def render_ues_connected(self, ax) -> None:
        time = np.arange(self.time)
        # ues_connected 表示在每个时间步上，有多少用户设备与基站保持连接
        ues_connected = self.monitor.scalar_results["number connected"]
        ax.plot(time, ues_connected, linewidth=1, color="black")

        ax.set_xlabel("Time")
        ax.set_ylabel("#Conn. UEs")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, len(self.userDict)])

    # 关闭仿真环境，同时终止其可视化的显示

    def close(self) -> None:
        pygame.quit()
        self.window = None
        self.closed = True
