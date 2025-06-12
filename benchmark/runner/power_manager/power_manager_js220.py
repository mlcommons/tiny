# power_manager_js220.py

from joulescope import scan_require_one

class JoulescopePowerManager:
    def __init__(self, **kwargs):
        self.js = scan_require_one()
        self.js.open()
        self._data_queue = []

    def power_on(self):
        self.js.parameters['sensor_power'] = 'on'

    def power_off(self):
        self.js.parameters['sensor_power'] = 'off'

    def start(self):
        self.js.stream_configure()
        self.js.stream_start()

    def stop(self):
        self.js.stream_stop()

    def get_results(self):
        for _, data in self.js.data_reader().items():
            self._data_queue.append(data['energy']['value'])
        return self._data_queue
