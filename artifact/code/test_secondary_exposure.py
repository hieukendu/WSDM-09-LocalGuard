"""Synthetic analytic fixtures, never benchmark or human results."""
import unittest

from cumulative_exposure import unique_disclosures
from secondary_exposure import cleared, weighted_record


class SecondaryTests(unittest.TestCase):
    def test_unique_not_sum(self):
        labels = [{"HEALTH"}, {"HEALTH", "FAMILY"}, set()]
        self.assertEqual(unique_disclosures([[0, 1], [0, 2], [1]], labels), (2, 2))
        self.assertEqual(unique_disclosures([[], []], labels), (0, 0))

    def test_severity_max_not_sum(self):
        self.assertEqual(weighted_record({"HEALTH", "FINANCE"}), 2)
        self.assertEqual(weighted_record({"FAMILY"}), 1)
        self.assertEqual(weighted_record(set()), 0)

    def test_grid_cap(self):
        scores = [.2] * 8
        self.assertEqual(cleared(scores, [-1, .2, 1]), [[], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
