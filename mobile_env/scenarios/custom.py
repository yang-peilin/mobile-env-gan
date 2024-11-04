from my_env_gan.core.base import MComCore
from my_env_gan.core.entities import BaseStation, UserEquipment
from my_env_gan.core.util import deep_dict_merge


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

        stations = [
            BaseStation(bs_id=0, pos=(50, 50), **env_config["bs"]),
            BaseStation(bs_id=1, pos=(50, 100), **env_config["bs"]),
            BaseStation(bs_id=2, pos=(100, 50), **env_config["bs"]),
            # BaseStation(bs_id=3, pos=(100, 100), **env_config["bs"]),
            # BaseStation(bs_id=4, pos=(150, 100), **env_config["bs"]),
            # BaseStation(bs_id=5, pos=(100, 150), **env_config["bs"]),
            # BaseStation(bs_id=6, pos=(150, 150), **env_config["bs"])
        ]

        users = [
            UserEquipment(ue_id=0, **env_config["ue"]),
            UserEquipment(ue_id=1, **env_config["ue"]),
            UserEquipment(ue_id=2, **env_config["ue"]),
            UserEquipment(ue_id=3, **env_config["ue"]),
            UserEquipment(ue_id=4, **env_config["ue"]),
            UserEquipment(ue_id=5, **env_config["ue"]),
            UserEquipment(ue_id=6, **env_config["ue"]),
            UserEquipment(ue_id=7, **env_config["ue"]),
            UserEquipment(ue_id=8, **env_config["ue"]),
        ]

        super().__init__(stations, users, config, render_mode)
