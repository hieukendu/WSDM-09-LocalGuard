"""Synthetic export guard tests, no sensitive data."""
import unittest
from build_supplement import check_aggregate, scan_text


class SupplementTests(unittest.TestCase):
    def test_reject_nested_identity(self):
        with self.assertRaises(ValueError):
            check_aggregate({"safe": [{"uid": "synthetic-fixture"}]})

    def test_reject_labels_and_text(self):
        for key in ["categories", "text", "prompt", "rec", "pidx", "examples"]:
            with self.assertRaises(ValueError):
                check_aggregate({key: []})

    def test_aggregates(self):
        check_aggregate({"task": "lamp3", "n_users": 450, "mean": .2, "bounds": [.1, .3]})

    def test_secrets_and_author_names(self):
        self.assertGreater(scan_text("hf_" + "x" * 30, []), 0)
        self.assertGreater(scan_text("Author: Example Person", ["Example Person"]), 0)
        self.assertEqual(scan_text("Anonymous aggregate result", ["Example Person"]), 0)


if __name__ == "__main__":
    unittest.main()
