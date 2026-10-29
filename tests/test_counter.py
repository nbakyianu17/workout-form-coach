import math
import unittest

from workout_form_coach import Phase, SquatCounter, knee_angle


class KneeAngleTests(unittest.TestCase):
    def test_straight_and_right_angles(self):
        self.assertAlmostEqual(
            knee_angle((0.0, 1.0), (0.0, 0.0), (0.0, -1.0)), 180.0
        )
        self.assertAlmostEqual(
            knee_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)), 90.0
        )

    def test_degenerate_or_non_finite_points_are_ignored(self):
        self.assertIsNone(knee_angle((0.0, 0.0), (0.0, 0.0), (0.0, 1.0)))
        self.assertIsNone(
            knee_angle((math.nan, 0.0), (0.0, 0.0), (0.0, 1.0))
        )


class SquatCounterTests(unittest.TestCase):
    def test_counts_only_a_complete_deep_squat(self):
        counter = SquatCounter()
        self.assertEqual(counter.update(175.0).phase, Phase.STANDING)
        self.assertEqual(counter.update(145.0).phase, Phase.DESCENDING)
        self.assertEqual(counter.update(95.0).phase, Phase.ASCENDING)
        completion = counter.update(170.0)
        self.assertTrue(completion.counted)
        self.assertEqual(completion.reps, 1)
        self.assertEqual(completion.phase, Phase.STANDING)

    def test_shallow_bend_does_not_count_and_resets(self):
        counter = SquatCounter()
        counter.update(140.0)
        reset = counter.update(170.0)
        self.assertEqual(reset.reps, 0)
        self.assertEqual(reset.phase, Phase.STANDING)
        self.assertFalse(reset.counted)

    def test_low_visibility_and_missing_angles_do_not_change_state(self):
        counter = SquatCounter()
        counter.update(145.0)
        before = counter.update(90.0, visibility=0.2)
        self.assertFalse(before.accepted)
        self.assertEqual(before.phase, Phase.DESCENDING)
        missing = counter.update(None)
        self.assertFalse(missing.accepted)
        self.assertEqual(missing.phase, Phase.DESCENDING)

    def test_stays_in_standing_until_threshold_crossed(self):
        counter = SquatCounter()
        for _ in range(10):
            update = counter.update(160.0)
        self.assertEqual(update.phase, Phase.STANDING)
        self.assertEqual(update.reps, 0)

    def test_invalid_configuration_and_observations_raise(self):
        with self.assertRaises(ValueError):
            SquatCounter(bottom_threshold=160.0, standing_threshold=160.0)
        with self.assertRaises(ValueError):
            SquatCounter(min_visibility=1.1)
        counter = SquatCounter()
        with self.assertRaises(ValueError):
            counter.update(181.0)
        with self.assertRaises(ValueError):
            counter.update(90.0, visibility=float("nan"))


if __name__ == "__main__":
    unittest.main()
