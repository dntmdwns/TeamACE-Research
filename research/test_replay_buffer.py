#!/usr/bin/env python3
"""Small deterministic tests for replay selection and quantization utilities."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay_buffer import (
    balanced_voxel_sample,
    dequantize_features_int8,
    quantize_features_int8,
    stratified_epoch_indices,
    weighted_voxel_draw,
    weighted_voxel_sample,
)


def test_voxel_round_robin():
    coords = torch.tensor([
        [0.01, 0.01, 0.01],
        [0.02, 0.01, 0.01],
        [1.01, 0.01, 0.01],
        [1.02, 0.01, 0.01],
        [2.01, 0.01, 0.01],
        [2.02, 0.01, 0.01],
    ])
    generator = torch.Generator().manual_seed(7)
    selected, voxel_count = balanced_voxel_sample(coords, 3, 0.5, generator)
    selected_voxels = torch.floor(coords[selected] / 0.5).to(torch.int64)
    assert voxel_count == 3
    assert len(torch.unique(selected_voxels, dim=0)) == 3


def _dense_and_sparse_pool():
    """One voxel with 950 points (x < 0.5) and 50 voxels with one point each (x = 1.01, 2.01, ...)."""
    dense = torch.rand(950, 3, generator=torch.Generator().manual_seed(3)) * 0.49
    sparse = torch.zeros(50, 3)
    sparse[:, 0] = torch.arange(1, 51) + 0.01
    return torch.cat((dense, sparse))


def test_weighted_voxel_sample_basics():
    coords = _dense_and_sparse_pool()
    for alpha in (0.0, 0.5, 1.0, 2.0):
        selected, voxel_count = weighted_voxel_sample(coords, 200, 0.5, torch.Generator().manual_seed(1), alpha)
        assert voxel_count == 51 and len(selected) == 200 and len(torch.unique(selected)) == 200
        again, _ = weighted_voxel_sample(coords, 200, 0.5, torch.Generator().manual_seed(1), alpha)
        assert torch.equal(selected, again)
        _, inverse = torch.unique(torch.floor(coords / 0.5).long(), dim=0, return_inverse=True)
        assert torch.equal(selected, weighted_voxel_draw(inverse, 200, torch.Generator().manual_seed(1), alpha))
    # K = N empties every voxel and returns a permutation of the pool.
    selected, _ = weighted_voxel_sample(coords, len(coords), 0.5, torch.Generator().manual_seed(2), 0.0)
    assert torch.equal(torch.sort(selected).values, torch.arange(len(coords)))
    for bad in (dict(alpha=-0.1), dict(alpha=float("nan")), dict(sample_count=1001), dict(voxel_size_m=0.0)):
        args = dict(scene_coords=coords, sample_count=10, voxel_size_m=0.5,
                    generator=torch.Generator().manual_seed(0), alpha=1.0)
        args.update(bad)
        try:
            weighted_voxel_sample(**args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid arguments {bad}")


def test_weighted_voxel_alpha_one_matches_random_and_zero_spreads():
    coords = _dense_and_sparse_pool()
    reps, k = 400, 50
    sparse_hits = {0.0: 0, 1.0: 0}
    inclusion = torch.zeros(len(coords))
    for rep in range(reps):
        for alpha in (0.0, 1.0):
            selected, _ = weighted_voxel_sample(coords, k, 0.5, torch.Generator().manual_seed(rep), alpha)
            sparse_hits[alpha] += int((selected >= 950).sum())
            if alpha == 1.0:
                inclusion[selected] += 1
    # alpha = 1: every point equally likely, as uniform sampling without replacement (expected 50 * 50/1000 =
    # 2.5 sparse points per draw; inclusion probability k/N = 0.05 for every point).
    assert abs(sparse_hits[1.0] / reps - 2.5) < 0.3, sparse_hits
    assert abs(inclusion[:950].mean() / reps - 0.05) < 0.002 and abs(inclusion[950:].mean() / reps - 0.05) < 0.01
    # alpha = 0: the dense voxel is one voxel among 51, so most of the 50 draws go to the single-point voxels.
    assert sparse_hits[0.0] / reps > 30, sparse_hits


def test_int8_round_trip():
    generator = torch.Generator().manual_seed(11)
    features = torch.randn((1000, 512), generator=generator, dtype=torch.float16)
    quantized, scales = quantize_features_int8(features)
    restored = dequantize_features_int8(quantized, scales)
    assert quantized.dtype == torch.int8
    assert scales.shape == (512,)
    assert torch.mean(torch.abs(restored.float() - features.float())).item() < 0.02


def test_quality_preserves_coverage_and_redistributes():
    # Negative coordinates must form a distinct voxel; the singleton is exhausted
    # after round one, so round two must draw from the other two voxels.
    coords = torch.tensor([[-0.01, 0., 0.], [0.01, 0., 0.], [0.02, 0., 0.],
                           [0.03, 0., 0.], [1.01, 0., 0.], [1.02, 0., 0.]])
    errors = torch.tensor([4., 3., 1., 2., 2., 0.5])
    selected, count = balanced_voxel_sample(
        coords, 5, 0.25, torch.Generator().manual_seed(7), quality=errors)
    assert count == 3
    assert set(selected.tolist()) == {0, 2, 3, 4, 5}
    assert len(torch.unique(torch.floor(coords[selected[:3]] / 0.25), dim=0)) == 3
    again, _ = balanced_voxel_sample(
        coords, 5, 0.25, torch.Generator().manual_seed(7), quality=errors)
    assert torch.equal(selected, again)
    for k in (1, 2, 6):
        chosen, _ = balanced_voxel_sample(
            coords, k, 0.25, torch.Generator().manual_seed(7), quality=errors)
        assert len(chosen) == len(torch.unique(chosen)) == k


def test_batch_quotas_and_epoch_size():
    batches = list(stratified_epoch_indices(
        4000000, 400000, 5120, 10, torch.Generator().manual_seed(11)))
    assert len(batches) == 781
    assert all(len(b) == 5120 and int((b >= 3600000).sum()) == 512 for b in batches)
    all_indices = torch.cat(batches)
    assert len(torch.unique(all_indices)) == len(all_indices)
    assert int(all_indices.min()) >= 0 and int(all_indices.max()) < 4000000
    assert len(list(stratified_epoch_indices(
        100, 10, 20, 10, torch.Generator().manual_seed(1)))) == 5
    try:
        list(stratified_epoch_indices(100, 10, 20, 14, torch.Generator()))
    except ValueError:
        pass
    else:
        raise AssertionError('Replay quota exceeding stored entries was accepted')


if __name__ == "__main__":
    test_voxel_round_robin()
    test_weighted_voxel_sample_basics()
    test_weighted_voxel_alpha_one_matches_random_and_zero_spreads()
    test_int8_round_trip()
    test_quality_preserves_coverage_and_redistributes()
    test_batch_quotas_and_epoch_size()
    print("Replay buffer tests passed.")
