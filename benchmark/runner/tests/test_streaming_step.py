import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class StreamingStepTests(unittest.TestCase):
    def test_streaming_step_with_simulated_board_responses(self):
        # script.py configures global logging and creates a session directory on
        # import. Isolate those side effects and never open a serial port.
        program = textwrap.dedent("""\
            from unittest.mock import Mock, patch
            from script import _ScriptStreamStep
            from streaming_ww_utils import summarize_sww_result

            dataset = Mock()
            dataset.get_file_by_index.return_value = {
                'wav_file': 'short_wav_easy10s.wav',
                'length_sec': 10.0,
                'detection_windows': [[1.0, 2.0]],
            }
            dut = Mock()
            dut.stop_detecting.return_value = [
                'target activations:', '[7, 8]', 'm-ready',
            ]
            interface = Mock()
            interface.print_detections.return_value = [
                'Detection Timestamps (ms)', '100,', '1350,', 'm-ready',
            ]
            interface.print_dutycycle.return_value = [
                'Duty cycle start times (s)', '1000, 2000, 3000,',
                'Duty cycle stop times (s)', '1500, 2500, 3500, m-ready',
            ]
            with patch('script.time.sleep'):
                result = _ScriptStreamStep().run(interface, dut, dataset, 'a')
                assert result['infer']['detections'] == [100, 1350]
                assert abs(result['dutycycle']['duty_cycle'] - 0.5) < 1e-12
                assert result['activations'] == [7, 8]
                interface.play_wave.assert_called_once_with(
                    'short_wav_easy10s.wav', timeout=20.0
                )
                summarize_sww_result([result], None)

                interface.print_detections.return_value = [
                    'Detection Timestamps (ms)', 'm-ready',
                ]
                interface.print_dutycycle.return_value = [
                    'Duty cycle start times (s)',
                    'Duty cycle stop times (s)', 'm-ready',
                ]
                try:
                    _ScriptStreamStep().run(interface, dut, dataset, 'a')
                except RuntimeError as error:
                    assert 'No complete processing pulses' in str(error), str(error)
                else:
                    raise AssertionError('Missing pulses must invalidate timing')

                interface.print_detections.return_value = []
                try:
                    _ScriptStreamStep().run(interface, dut, dataset, 'a')
                except ValueError as error:
                    assert 'Empty detection timestamp response' in str(error), str(error)
                else:
                    raise AssertionError('An empty response must be diagnosed')
            print('Streaming integration checks passed')
        """)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Streaming integration checks passed", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
