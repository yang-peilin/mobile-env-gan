from typing import Dict

import pandas as pd


class Monitor:
    def __init__(
        self, scalar_metrics: Dict, ue_metrics: Dict, bs_metrics: Dict, **kwargs
    ):
        # scalar_metrics：标量指标字典，用于跟踪与仿真整体有关的数值（例如平均数据速率、总连接数）
        self.scalar_metrics: Dict = scalar_metrics
        # ue_metrics：用户设备(UE)的指标字典，用于跟踪各个用户设备的指标（例如数据速率、信号质量）
        self.ue_metrics: Dict = ue_metrics
        # bs_metrics：基站(BS)的指标字典，用于跟踪各个基站的指标（例如服务的用户数、带宽使用率）
        self.bs_metrics: Dict = bs_metrics

        self.scalar_results: Dict = None
        self.ue_results: Dict = None
        self.bs_results: Dict = None

    def reset(self):
        self.scalar_results = {name: [] for name in self.scalar_metrics}
        self.ue_results = {name: [] for name in self.ue_metrics}
        self.bs_results = {name: [] for name in self.bs_metrics}

    def update(self, simulation):
        # 计算标量、UE、BS 的指标
        scalar_updates = {
            name: metric(simulation) for name, metric in self.scalar_metrics.items()
        }
        ue_updates = {
            name: metric(simulation) for name, metric in self.ue_metrics.items()
        }
        bs_updates = {
            name: metric(simulation) for name, metric in self.bs_metrics.items()
        }

        # 更新结果，通过添加新计算的值到对应的结果列表
        self.scalar_results = {
            name: self.scalar_results[name] + [scalar_updates[name]] for name in self.scalar_metrics
        }
        self.ue_results = {
            name: self.ue_results[name] + [ue_updates[name]] for name in self.ue_metrics
        }
        self.bs_results = {
            name: self.bs_results[name] + [bs_updates[name]] for name in self.bs_metrics
        }

    def load_results(self):
        # 加载标量结果
        scalar_results = pd.DataFrame(self.scalar_results)
        scalar_results.index.names = ["Time Step"]

        # 构建ue_results字典：提取每个时间步内每个用户设备的指标值，构建以(metric, ue_id)为键的数据结构
        ue_results = {
            (metric, ue_id): [values.get(ue_id) for values in entries]
            for metric, entries in self.ue_results.items()
            for ue_id in set().union(*entries)
        }
        ue_results = pd.DataFrame(ue_results).transpose()
        ue_results.index.names = ["Metric", "UE ID"]
        # 将某一层次的列索引转为行索引
        ue_results = ue_results.stack()
        ue_results.index.names = ["Metric", "UE ID", "Time Step"]
        ue_results = ue_results.reorder_levels(["Time Step", "UE ID", "Metric"])
        ue_results = ue_results.unstack()

        # 加载 BS 结果
        bs_results = {
            (metric, bs_id): [values.get(bs_id) for values in entries]
            for metric, entries in self.bs_results.items()
            for bs_id in set().union(*entries)
        }
        bs_results = pd.DataFrame(bs_results).transpose()
        # 设置数据帧行索引的名称，以使数据更具可读性
        bs_results.index.names = ["Metric", "BS ID"]
        bs_results = bs_results.stack()
        bs_results.index.names = ["Metric", "BS ID", "Time Step"]
        bs_results = bs_results.reorder_levels(["Time Step", "BS ID", "Metric"])
        bs_results = bs_results.unstack()

        # 最终返回三个数据帧，分别代表标量、UE和BS的指标结果
        return scalar_results, ue_results, bs_results

    # 对于每个指标，返回最新（最后一个时间步）的值，构建一个包含标量、UE和BS指标的字典
    def info(self):
        # 如果没有标量结果，则返回空字典
        if any(len(results) == 0 for results in self.scalar_results.values()):
            return {}

        scalar_info = {name: values[-1] for name, values in self.scalar_results.items()}
        ue_info = {name: values[-1] for name, values in self.ue_results.items()}
        bs_info = {name: values[-1] for name, values in self.bs_results.items()}

        return {**scalar_info, **ue_info, **bs_info}
