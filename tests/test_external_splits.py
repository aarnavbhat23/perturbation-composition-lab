import unittest
from replication import gene_holdout
from wessels_prepare import condition


class ExternalPartitions(unittest.TestCase):
    def test_gene_holdout_retains_validation_and_known_single_setting(self):
        pairs = [
            "A+B",
            "A+C",
            "B+C",
            "B+D",
            "C+D",
            "E+F",
            "E+G",
            "F+G",
            "H+I",
            "H+J",
            "I+J",
            "K+L",
        ]
        labels = gene_holdout(pairs, "A")
        self.assertEqual(set(labels), {"train", "test", "validation"})
        self.assertTrue(
            all("A" in p.split("+") for p, s in zip(pairs, labels) if s == "test")
        )
        self.assertTrue(
            all("A" not in p.split("+") for p, s in zip(pairs, labels) if s != "test")
        )

    def test_explicit_nontargeting_and_pair_order(self):
        self.assertEqual(condition("NT_NT"), "ctrl")
        self.assertEqual(condition("MYB_NT"), "MYB")
        self.assertEqual(condition("NT_MYB"), "MYB")
        self.assertEqual(condition("MYB_EP300"), "EP300+MYB")
        self.assertIsNone(condition("MYB_MYB"))
