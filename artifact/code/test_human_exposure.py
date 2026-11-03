"""Exhaustive synthetic tests of missing-human-label bounds, not human results."""
import itertools
import unittest

import numpy as np

from crc import crc_index, monotonize
from human_exposure import fixed_policy_summary, index_bounds, loss_bounds


class ExposureBoundsTests(unittest.TestCase):
    def test_unknown_is_interval(self):
        sets = [[[], [0], [0, 1]]]
        low, high = loss_bounds(sets, [{0: 1}])
        self.assertEqual(low.tolist(), [[0, 1, 1]])
        self.assertEqual(high.tolist(), [[0, 1, 2]])
        summary = fixed_policy_summary(sets, [{0: 1}], [2])
        self.assertFalse(summary["exact"])
        self.assertEqual(summary["human_labelled_admitted_records"], 1)

    def test_full_labels_exact(self):
        sets = [[[], [0], [0, 1]]] * 10
        lo, hi = loss_bounds(sets, [{0: 1, 1: 0}] * 10)
        self.assertTrue(np.array_equal(lo, hi))
        self.assertEqual(*index_bounds(lo, hi, .5, 2))

    def test_every_completion_is_inside_index_interval(self):
        # Non-nested cleared sets require the running-maximum envelope.
        sets = [[[], [0], [1], [0, 2]]] * 3
        partial = [{0: 0}, {1: 1}, {2: 0}]
        unknown = [(u, i) for u in range(3) for i in range(3) if i not in partial[u]]
        low, high = loss_bounds(sets, partial)
        for bits in itertools.product([0, 1], repeat=len(unknown)):
            complete = [dict(x) for x in partial]
            for (u, i), value in zip(unknown, bits):
                complete[u][i] = value
            truth, _ = loss_bounds(sets, complete)
            for alpha in [.2, .5, 1, 1.5, 2]:
                left, right = index_bounds(low, high, alpha, 2)
                actual = crc_index(monotonize(truth), alpha, 2)
                self.assertLessEqual(left, actual)
                self.assertLessEqual(actual, right)

    def test_invalid_sets_and_labels(self):
        for sets, labels in [([[[], [0, 0]]], [{}]), ([[[0], [1]]], [{}]),
                             ([[[], [0]]], [{0: None}])]:
            with self.assertRaises(ValueError):
                loss_bounds(sets, labels)

    def test_invalid_bounds(self):
        for lo, hi in [([[0, 1]], [[0, 0]]), ([[0, 0]], [[0, 3]]),
                       ([[1, 1]], [[1, 1]]), ([[0, 0]], [[0, float('nan')]])]:
            with self.assertRaises(ValueError):
                index_bounds(lo, hi, .2, 2)


if __name__ == "__main__":
    unittest.main()
