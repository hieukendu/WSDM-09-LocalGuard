"""Synthetic unit tests only; these are not human annotation results."""
import unittest

from human_audit import agreement, parse_labels, precision_recall


class HumanAuditTests(unittest.TestCase):
    def test_missing_not_negative(self):
        with self.assertRaises(ValueError):
            parse_labels([dict(audit_id="x", human_discloses="", human_categories="")], {"x"})

    def test_inconsistent_or_unknown(self):
        for binary, cats in [("0", "HEALTH"), ("1", ""), ("1", "UNKNOWN")]:
            with self.assertRaises(ValueError):
                parse_labels([dict(audit_id="x", human_discloses=binary, human_categories=cats)], {"x"})

    def test_duplicate_and_incomplete(self):
        row = dict(audit_id="x", human_discloses="0", human_categories="")
        for rows, ids in [([row, row], {"x"}), ([row], {"x", "y"})]:
            with self.assertRaises(ValueError):
                parse_labels(rows, ids)

    def test_multilabel(self):
        self.assertEqual(parse_labels([dict(audit_id="x", human_discloses="1", human_categories="HEALTH;FAMILY")], {"x"}),
                         {"x": {"HEALTH", "FAMILY"}})

    def test_undefined_kappa(self):
        self.assertIsNone(agreement([0, 0], [0, 0])["kappa"])
        self.assertEqual(agreement([0, 1], [0, 1])["kappa"], 1)

    def test_confusion(self):
        r = precision_recall([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual([r[k] for k in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 1])
        self.assertEqual(r["precision"], .5)
        self.assertIsNone(precision_recall([0], [0])["recall"])


if __name__ == "__main__":
    unittest.main()
