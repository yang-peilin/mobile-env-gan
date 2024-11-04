from typing import Tuple

from shapely.geometry import Point


class BaseStation:
    def __init__(
        self,
        bs_id: int,                 # 基站的唯一标识符 (int 类型)
        pos: Tuple[float, float],   # pos：基站的位置 (Tuple[float, float])，表示基站的坐标 (x, y)
        bw: float,                  # 基站的带宽 (float)，表示基站可以使用的频谱宽度（单位 Hz）
        freq: float,                # 基站的工作频率 (float)，表示基站工作在哪个频率（单位 MHz）
        tx: float,                  # 基站的发射功率 (float)，表示基站发出的信号强度（单位 dBm）
        height: float,              # 基站的高度 (float)，单位是米
    ):
        # Python3.7不支持Final类型标注 这个项目需要支持python3.7
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        self.height = height  # in m

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"BS: {self.bs_id}"


class UserEquipment:
    def __init__(
        self,
        ue_id: int,         # 用户设备的唯一标识符 (int 类型)
        velocity: float,    # 用户设备的速度 (float)，单位可以是米/秒
        snr_tr: float,      # 信噪比门限 (float)，用于判断用户设备与基站之间的连接质量是否足够好
        noise: float,       # 用户设备的接收端噪声功率 (float)
        height: float,      # 用户设备的高度 (float)，单位是米
    ):
        self.ue_id = ue_id
        self.velocity: float = velocity
        self.snr_threshold = snr_tr
        self.noise = noise      # 用户设备的到达时间（开始时间）
        self.height = height    # 用户设备的离开时间，表示设备何时离开网络

        self.x: float = None
        self.y: float = None
        self.stime: int = None
        self.extime: int = None

    @property
    def point(self):
        return Point(int(self.x), int(self.y))

    def __str__(self):
        return f"UE: {self.ue_id}"
