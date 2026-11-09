"""Configuration parsing tests; no device or optional driver is needed."""
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from runner_config import parse_device_config, parse_test_script


class RunnerConfigTest(unittest.TestCase):
    def test_inline_and_file_device_config_match(self):
        raw = 'devices:\n- type: dut\n  baud: 115200\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'devices.yaml'
            path.write_text(raw)
            self.assertEqual(parse_device_config(path, None),
                             parse_device_config(None, raw))
            self.assertEqual(parse_device_config(None, raw)['devices'][0]['baud'], 115200)

    def test_shipped_scripts_remain_readable(self):
        for path in Path(__file__).parent.glob('tests_*.yaml'):
            with self.subTest(path=path.name):
                self.assertIsInstance(parse_test_script(path), dict)

    def test_all_inputs_reject_python_tags_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'must-not-exist'
            raw = '!!python/object/apply:os.mkdir [' + json.dumps(str(marker)) + ']\n'
            path = Path(directory) / 'config.yaml'
            path.write_text(raw)
            for parser in [lambda: parse_device_config(None, raw),
                           lambda: parse_device_config(path, None),
                           lambda: parse_test_script(path)]:
                with self.subTest(parser=parser), self.assertRaises(yaml.constructor.ConstructorError):
                    parser()
                self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
