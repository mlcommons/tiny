# power_manager.py

from power_manager.power_manager_lpm import LPMManager
from power_manager.power_manager_js220 import JoulescopePowerManager

class PowerManager:
    def __new__(cls, name, **kwargs):
        if name == 'lpm01a':
            return LPMManager(**kwargs)
        elif name == 'js220':
            return JoulescopePowerManager(**kwargs)
        else:
            raise ValueError(f"Unsupported power device: {name}")
