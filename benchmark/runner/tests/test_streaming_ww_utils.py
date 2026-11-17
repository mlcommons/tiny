import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streaming_ww_utils as sww


def dutycycle_response(starts=(), stops=()):
    # Match InterfaceMenu::PrintDutycycle, including a terminator on the
    # same line as the last stop timestamp.
    return [
        "Duty cycle start times (s)",
        ", ".join(map(str, starts)),
        "Duty cycle stop times (s)",
        ", ".join(map(str, stops)) + ", m-ready",
    ]


class StreamingResponseTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_empty_detection_response_has_context(self):
        with self.assertRaisesRegex(ValueError, "Empty detection timestamp response"):
            sww.process_timestamps([])

    def test_framed_response_with_no_detections_is_preserved(self):
        self.assertEqual(
            sww.process_timestamps(["Detection Timestamps (ms)", "m-ready"]),
            [],
        )

    def test_detection_timestamps_are_preserved(self):
        self.assertEqual(
            sww.process_timestamps(
                ["Detection Timestamps (ms)", "100,", "1350,", "m-ready"]
            ),
            [100, 1350],
        )

    def test_malformed_detection_responses_are_rejected(self):
        for response in (
            ["unexpected", "m-ready"],
            ["Detection Timestamps (ms)", "100,"],
            ["Detection Timestamps (ms)", "invalid,", "m-ready"],
        ):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    sww.process_timestamps(response)

    def test_empty_dutycycle_response_has_context(self):
        with self.assertRaisesRegex(ValueError, "Empty duty cycle response"):
            sww.process_dutycycle([])

    def test_no_processing_pulses_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "No complete processing pulses.*PROCESSING"):
            sww.process_dutycycle(dutycycle_response())

    def test_one_processing_pulse_cannot_establish_a_period(self):
        with self.assertRaisesRegex(RuntimeError, "at least two complete processing pulses"):
            sww.process_dutycycle(dutycycle_response([1000], [1500]))

    def test_unmatched_edges_are_checked_after_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "at least two complete processing pulses"):
            sww.process_dutycycle(dutycycle_response([1000, 2000], [1500]))

    def test_no_complete_pulses_are_reported_after_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "No complete processing pulses"):
            sww.process_dutycycle(dutycycle_response([1000], []))

    def test_valid_dutycycle_measurements_are_preserved(self):
        result = sww.process_dutycycle(
            dutycycle_response([1000, 2000, 3000], [1500, 2500, 3500])
        )
        self.assertAlmostEqual(result["period"], 0.01)
        self.assertAlmostEqual(result["processing_time"], 0.005)
        self.assertAlmostEqual(result["duty_cycle"], 0.5)
        for actual, expected in zip(result["start_times_s"], [0.01, 0.02, 0.03]):
            self.assertAlmostEqual(actual, expected)

    def test_two_complete_pulses_are_sufficient(self):
        result = sww.process_dutycycle(dutycycle_response([1000, 2000], [1500, 2500]))
        self.assertAlmostEqual(result["period"], 0.01)
        self.assertAlmostEqual(result["duty_cycle"], 0.5)

    def test_existing_truncation_keeps_complete_measurements(self):
        result = sww.process_dutycycle(
            dutycycle_response([1000, 2000, 3000], [1500, 2500])
        )
        self.assertEqual(len(result["start_times_s"]), 2)
        self.assertAlmostEqual(result["duty_cycle"], 0.5)

    def test_malformed_dutycycle_framing_is_rejected(self):
        for response in (
            ["unexpected", "m-ready"],
            ["Duty cycle start times (s)", "Duty cycle stop times (s)"],
        ):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    sww.process_dutycycle(response)

    def test_missing_detection_sync_pulse_is_not_scored_as_zero_accuracy(self):
        with self.assertRaisesRegex(RuntimeError, "initial synchronization pulse"):
            sww.calc_detection_stats([], [[1.0, 2.0]])

    def test_sync_pulse_only_counts_wakewords_as_missed(self):
        true_pos, false_neg, false_pos = sww.calc_detection_stats([100], [[1.0, 2.0]])
        self.assertEqual(true_pos, [])
        self.assertEqual(false_neg, [1.0])
        self.assertEqual(false_pos.tolist(), [])

    def test_valid_detection_scoring_is_preserved(self):
        true_pos, false_neg, false_pos = sww.calc_detection_stats(
            [100, 1350, 5100], [[1.0, 2.0]]
        )
        self.assertEqual(true_pos, [1.0])
        self.assertEqual(false_neg, [])
        self.assertEqual(false_pos.tolist(), [5.0])


if __name__ == "__main__":
    unittest.main()
