from typing import Dict

import pandas as pd


class Monitor:
    def __init__(
        self, scalar_metrics: Dict, ue_metrics: Dict, bs_metrics: Dict, **kwargs
    ):
        self.scalar_metrics: Dict = scalar_metrics
        self.ue_metrics: Dict = ue_metrics
        self.bs_metrics: Dict = bs_metrics

        self.scalar_results: Dict = None
        self.ue_results: Dict = None
        self.bs_results: Dict = None

    def reset(self):
        self.scalar_results = {name: [] for name in self.scalar_metrics}
        self.ue_results = {name: [] for name in self.ue_metrics}
        self.bs_results = {name: [] for name in self.bs_metrics}

    def update(self, simulation):
        scalar_updates = {
            name: metric(simulation) for name, metric in self.scalar_metrics.items()
        }
        ue_updates = {
            name: metric(simulation) for name, metric in self.ue_metrics.items()
        }
        bs_updates = {
            name: metric(simulation) for name, metric in self.bs_metrics.items()
        }

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
        scalar_results = pd.DataFrame(self.scalar_results)
        scalar_results.index.names = ["Time Step"]

        ue_results = {
            (metric, ue_id): [values.get(ue_id) for values in entries]
            for metric, entries in self.ue_results.items()
            for ue_id in set().union(*entries)
        }
        ue_results = pd.DataFrame(ue_results).transpose()
        ue_results.index.names = ["Metric", "UE ID"]
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
