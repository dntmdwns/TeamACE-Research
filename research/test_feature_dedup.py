#!/usr/bin/env python3
"""Selector invariants and exact comparison to a simple greedy reference."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_buffer import voxel_feature_dedup_sample


class DedupTest(unittest.TestCase):
    def select(self, coords, features, k, **kwargs):
        return voxel_feature_dedup_sample(coords, features, k, 0.25,
                                         torch.Generator().manual_seed(17), **kwargs)

    def test_duplicate_only_within_voxel_and_conservative_zero(self):
        coords = torch.tensor([[0., 0., 0.]] * 4 + [[1., 0., 0.]])
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 0.], [1., 0.]])
        selected, meta = self.select(coords, features, 4, block_size=1)
        self.assertEqual(set(selected.tolist()) & {2, 3, 4}, {2, 3, 4})
        self.assertEqual(len(set(selected.tolist()) & {0, 1}), 1)
        self.assertEqual(meta['dedup_rejected'], 1)
        self.assertEqual(meta['dedup_backfill_count'], 0)
        self.assertEqual(meta['dedup_zero_norm_candidates'], 1)

    def test_blocked_matches_reference_and_index_mapping(self):
        rng = torch.Generator().manual_seed(13)
        features = torch.randn(90, 8, generator=rng)
        features[20:40] = features[:20]
        coords = torch.zeros(70, 3)
        coords[40:, 0] = -0.01  # negative coordinates must be a separate voxel
        mapping = torch.arange(10, 80)
        k, threshold = 15, 0.7
        selected, meta = self.select(coords, features, k, cosine_threshold=threshold,
                                     block_size=3, feature_indices=mapping)
        again, _ = self.select(coords, features[mapping], k, cosine_threshold=threshold,
                               block_size=11)
        self.assertTrue(torch.equal(selected, again))
        shortlist = torch.randperm(70, generator=torch.Generator().manual_seed(17))[:k * 4]
        vectors = torch.nn.functional.normalize(features[mapping[shortlist]], dim=1)
        voxels = torch.floor(coords[shortlist] / 0.25).long()
        kept = []
        for i in range(len(shortlist)):
            if not any(torch.equal(voxels[i], voxels[j]) and
                       float(vectors[i] @ vectors[j]) >= threshold for j in kept):
                kept.append(i)
        self.assertEqual(meta['dedup_survivors'], len(kept))
        self.assertTrue(set(selected.tolist()) <= set(shortlist[kept].tolist()))
        self.assertEqual(len(selected.unique()), k)

    def test_backfill_and_determinism(self):
        coords, features = torch.zeros(20, 3), torch.ones(20, 4)
        selected, meta = self.select(coords, features, 10, block_size=3)
        again, same_meta = self.select(coords, features, 10, block_size=3)
        self.assertTrue(torch.equal(selected, again))
        self.assertEqual(meta, same_meta)
        self.assertEqual(meta['dedup_survivors'], 1)
        self.assertEqual(meta['dedup_backfill_count'], 9)
        self.assertEqual(len(selected.unique()), 10)

    def test_invalid_inputs(self):
        coords, features = torch.zeros(5, 3), torch.ones(5, 2)
        for kwargs in ({'cosine_threshold': float('nan')}, {'cosine_threshold': 0},
                       {'pool_factor': 0}, {'block_size': 0}):
            with self.assertRaises(ValueError):
                self.select(coords, features, 2, **kwargs)
        features[0] = float('nan')
        with self.assertRaises(ValueError):
            self.select(coords, features, 5)


if __name__ == '__main__':
    unittest.main()
