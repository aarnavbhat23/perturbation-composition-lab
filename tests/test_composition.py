import unittest
import numpy as np
import torch
from pseudobulk import canonical_condition
from composer import ResidualComposer
from benchmark import split_pairs, metrics


class CompositionTests(unittest.TestCase):
    def test_order_is_not_a_new_condition(self):
        self.assertEqual(canonical_condition("B+A"), "A+B")
        self.assertEqual(canonical_condition("ctrl+A"), "A")
        self.assertEqual(canonical_condition("ctrl"), "ctrl")

    def test_duplicate_gene_rejected(self):
        with self.assertRaises(ValueError):
            canonical_condition("A+A")

    def test_model_is_symmetric(self):
        torch.manual_seed(4)
        model = ResidualComposer(4).eval()
        a, b = torch.randn(6, 4), torch.randn(6, 4)
        torch.testing.assert_close(model(a, b), model(b, a), atol=0, rtol=0)

    def test_split_never_uses_cells(self):
        pairs = [f"A+{i}" for i in range(100)]
        split = split_pairs(pairs)
        self.assertEqual((split == "test").sum(), 20)
        np.testing.assert_array_equal(split, split_pairs(pairs))

    def test_perfect_metrics(self):
        x = np.arange(40).reshape(2, 20)
        self.assertEqual(metrics(x, x, x)[0]["all_gene_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
