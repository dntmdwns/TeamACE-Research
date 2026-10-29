"""CPU checks for the exact greedy XYZ FPS rule and tuple uniqueness."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fps_sampling import farthest_point_sample


class FPSTest(unittest.TestCase):
    def test_greedy_rule_and_reproducibility(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [4., 0., 0.],
                            [4., 3., 0.], [2., 2., 0.], [0., 0., 0.]])
        def select():
            return farthest_point_sample(xyz, 5, torch.Generator().manual_seed(19))
        indices = select()
        self.assertTrue(torch.equal(indices, select()))
        self.assertEqual(len(indices.unique()), 5)
        for i in range(1, len(indices)):
            distances = torch.cdist(xyz, xyz[indices[:i]]).square().min(dim=1).values
            distances[indices[:i]] = -1
            self.assertAlmostEqual(float(distances[indices[i]]), float(distances.max()), places=5)

    def test_identical_coordinates_select_distinct_tuples(self):
        indices = farthest_point_sample(torch.zeros(12, 3), 12, torch.Generator().manual_seed(2))
        self.assertEqual(len(indices.unique()), 12)

    def test_invalid_input(self):
        for xyz, k in [(torch.zeros(4, 2), 2), (torch.zeros(4, 3), 0),
                       (torch.zeros(4, 3), 5), (torch.full((4, 3), float('nan')), 2)]:
            with self.assertRaises(ValueError):
                farthest_point_sample(xyz, k, torch.Generator().manual_seed(2))


if __name__ == '__main__':
    unittest.main()
