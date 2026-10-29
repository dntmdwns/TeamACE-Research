"""Utilities for compact, coverage-aware ACE feature replay buffers."""

import torch
import math


@torch.no_grad()
def voxel_feature_dedup_sample(scene_coords, features, sample_count, voxel_size_m,
                               generator, cosine_threshold=0.98, pool_factor=4,
                               block_size=128, feature_indices=None):
    """Random shortlist, then greedy cosine dedup within each predicted XYZ voxel.

    No equal per-voxel quota. Returns exactly K unique candidate indices: when
    dedup leaves fewer than K survivors, randomly backfill rejected candidates
    and report this explicitly. Similarity ignores feature magnitude and camera
    pose; it is a heuristic for redundant observations, not semantic identity.
    Only the bounded shortlist is examined. CPU RNG controls all ordering.
    """
    if (scene_coords.ndim != 2 or scene_coords.shape[1] != 3 or
            features.ndim != 2 or features.shape[1] < 1 or
            (feature_indices is None and len(features) != len(scene_coords))):
        raise ValueError('Expected aligned Nx3 coordinates and NxD features')
    if feature_indices is not None:
        feature_indices = feature_indices.detach().cpu()
        if (feature_indices.shape != (len(scene_coords),) or feature_indices.dtype != torch.long or
                int(feature_indices.min()) < 0 or int(feature_indices.max()) >= len(features)):
            raise ValueError('Invalid mapping from candidates to feature rows')
    if (not isinstance(sample_count, int) or not 1 <= sample_count <= len(scene_coords) or
            not math.isfinite(voxel_size_m) or voxel_size_m <= 0 or
            not math.isfinite(cosine_threshold) or not 0 < cosine_threshold <= 1 or
            not isinstance(pool_factor, int) or pool_factor < 1 or
            not isinstance(block_size, int) or block_size < 1):
        raise ValueError('Invalid dedup size, voxel size, threshold, pool factor, or block size')
    coords = scene_coords.detach().cpu()
    if not torch.isfinite(coords).all():
        raise ValueError('Coordinates must be finite')
    n = min(len(coords), sample_count * pool_factor)
    shortlist = torch.randperm(len(coords), generator=generator)[:n]
    # Only shortlist features are copied/normalized, never the full training buffer.
    rows = shortlist if feature_indices is None else feature_indices[shortlist]
    vectors = features[rows.to(features.device)].float()
    if not torch.isfinite(vectors).all():
        raise ValueError('Shortlisted features must be finite')
    norms = torch.linalg.vector_norm(vectors, dim=1)
    nonzero = (norms > 1e-12).cpu()
    vectors = vectors / norms.clamp_min(1e-12).unsqueeze(1)
    _, inverse = torch.unique(torch.floor(coords[shortlist] / voxel_size_m).long(),
                              dim=0, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    groups = torch.split(order, torch.bincount(inverse).tolist())
    survivors, rejected = [], []
    for group in groups:
        accepted = []
        for start in range(0, len(group), block_size):
            block = group[start:start + block_size]
            query = vectors[block.to(vectors.device)]
            duplicate = torch.zeros(len(block), dtype=torch.bool, device=vectors.device)
            # Bound temporary similarity matrices even if a voxel is very dense.
            for ref_start in range(0, len(accepted), block_size):
                refs = torch.tensor(accepted[ref_start:ref_start + block_size], device=vectors.device)
                duplicate |= (query @ vectors[refs].T >= cosine_threshold).any(dim=1)
            within = (query @ query.T >= cosine_threshold).cpu()
            blocked = duplicate.cpu()
            for j, index in enumerate(block.tolist()):
                # Zero vectors have no defined cosine; retain conservatively.
                if nonzero[index] and blocked[j]:
                    rejected.append(index)
                else:
                    accepted.append(index)
                    if nonzero[index]:
                        blocked[j + 1:] |= within[j, j + 1:]
        survivors.extend(accepted)
    survivor_count = len(survivors)
    survivors = torch.tensor(survivors, dtype=torch.long)
    survivors = survivors[torch.randperm(len(survivors), generator=generator)]
    backfill = max(0, sample_count - len(survivors))
    if backfill:
        rejected_tensor = torch.tensor(rejected, dtype=torch.long)
        fill = rejected_tensor[torch.randperm(len(rejected_tensor), generator=generator)[:backfill]]
        selected = torch.cat((survivors, fill))
    else:
        selected = survivors[:sample_count]
    selected = selected[torch.randperm(len(selected), generator=generator)]
    return shortlist[selected], {
        'dedup_cosine_threshold': cosine_threshold, 'dedup_pool_factor': pool_factor,
        'dedup_shortlist_candidates': n, 'dedup_survivors': survivor_count,
        'dedup_rejected': len(rejected), 'dedup_backfill_count': backfill,
        'dedup_backfill_fraction': backfill / sample_count,
        'dedup_zero_norm_candidates': int((~nonzero).sum()),
        'dedup_block_size': block_size,
    }


INT8_ENCODING = "int8_symmetric_per_channel"
FP16_ENCODING = "fp16"
FP32_ENCODING = "fp32"


def balanced_voxel_sample(scene_coords, sample_count, voxel_size_m, generator, quality=None):
    """Select indices in voxel-balanced rounds with random tie breaking.

    Every occupied voxel contributes one randomly chosen point before any voxel
    contributes a second point. Additional rounds continue until the requested
    sample count is reached. Inputs and returned indices live on the CPU.
    If quality is supplied, pick lower-error points first within each voxel.
    """
    if scene_coords.ndim != 2 or scene_coords.shape[1] != 3:
        raise ValueError("scene_coords must have shape Nx3")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if len(scene_coords) < sample_count:
        raise ValueError("Not enough valid scene coordinates for the requested sample count")

    scene_coords = scene_coords.detach().cpu()
    voxel_coords = torch.floor(scene_coords / voxel_size_m).to(torch.int64)
    _, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
    voxel_count = int(inverse.max().item()) + 1

    # Randomize points within each voxel, then group them with a stable sort.
    random_point_order = torch.randperm(len(scene_coords), generator=generator)
    if quality is not None:
        if quality.shape != (len(scene_coords),) or not torch.isfinite(quality).all():
            raise ValueError("quality must contain one finite error per candidate")
        # Lower error first; the initial permutation breaks exact ties reproducibly.
        random_point_order = random_point_order[
            torch.argsort(quality.cpu()[random_point_order], stable=True)
        ]
    shuffled_inverse = inverse[random_point_order]
    group_order = torch.argsort(shuffled_inverse, stable=True)
    grouped_indices = random_point_order[group_order]
    grouped_voxels = shuffled_inverse[group_order]

    counts = torch.bincount(grouped_voxels, minlength=voxel_count)
    starts = torch.cumsum(counts, dim=0) - counts
    ranks_within_voxel = torch.arange(len(scene_coords)) - torch.repeat_interleave(starts, counts)

    # Randomize the order of voxels within every selection round.
    voxel_order = torch.randperm(voxel_count, generator=generator)
    voxel_priority = torch.empty(voxel_count, dtype=torch.int64)
    voxel_priority[voxel_order] = torch.arange(voxel_count)
    priority = ranks_within_voxel * voxel_count + voxel_priority[grouped_voxels]
    chosen_positions = torch.topk(priority, sample_count, largest=False, sorted=True).indices

    return grouped_indices[chosen_positions], voxel_count


def weighted_voxel_sample(scene_coords, sample_count, voxel_size_m, generator, alpha=1.0):
    """Select indices by drawing voxels with probability proportional to (remaining candidates)^alpha.

    Each draw picks a voxel, then a random not-yet-chosen candidate inside it; a voxel drops out once all of
    its candidates are chosen. Draws repeat until sample_count indices are chosen, without replacement.
    alpha = 0 gives every non-empty voxel the same weight; alpha = 1 gives every remaining candidate the same
    probability, i.e. the same distribution as uniform sampling without replacement from the whole pool.
    Inputs and returned indices live on the CPU.
    """
    if scene_coords.ndim != 2 or scene_coords.shape[1] != 3:
        raise ValueError("scene_coords must have shape Nx3")
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if len(scene_coords) < sample_count:
        raise ValueError("Not enough valid scene coordinates for the requested sample count")

    scene_coords = scene_coords.detach().cpu()
    voxel_coords = torch.floor(scene_coords / voxel_size_m).to(torch.int64)
    _, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
    voxel_count = int(inverse.max().item()) + 1
    return weighted_voxel_draw(inverse, sample_count, generator, alpha), voxel_count


def weighted_voxel_draw(inverse, sample_count, generator, alpha):
    """The draws of weighted_voxel_sample, given each candidate's voxel id (0..V-1, every id occupied)."""
    voxel_count = int(inverse.max().item()) + 1
    # Randomize points within each voxel, then group them with a stable sort: taking a voxel's next point in
    # this order is a uniform draw among its remaining candidates.
    random_point_order = torch.randperm(len(inverse), generator=generator)
    grouped_indices = random_point_order[torch.argsort(inverse[random_point_order], stable=True)]
    counts = torch.bincount(inverse, minlength=voxel_count).tolist()
    starts = [0] * voxel_count
    for v in range(1, voxel_count):
        starts[v] = starts[v - 1] + counts[v - 1]

    taken = [0] * voxel_count
    weights = torch.tensor([float(c) ** alpha for c in counts], dtype=torch.float64)
    uniforms = torch.rand(sample_count, generator=generator, dtype=torch.float64).tolist()
    chosen = torch.empty(sample_count, dtype=torch.int64)
    for i in range(sample_count):
        # Inverse-CDF draw: the first voxel whose cumulative weight exceeds u * total (never an empty voxel).
        cumulative = torch.cumsum(weights, dim=0)
        target = torch.tensor([uniforms[i] * float(cumulative[-1])], dtype=torch.float64)
        v = min(int(torch.searchsorted(cumulative, target, right=True)), voxel_count - 1)
        while weights[v] == 0:   # guards the clamp above against rounding at the top end
            v -= 1
        chosen[i] = grouped_indices[starts[v] + taken[v]]
        taken[v] += 1
        remaining = counts[v] - taken[v]
        weights[v] = float(remaining) ** alpha if remaining else 0.0
    return chosen


def stratified_epoch_indices(total, replay_count, batch_size, ratio, generator):
    """Yield fixed-size batches with exact replay quotas, without replacement per epoch."""
    old_per_batch = round(batch_size * ratio / 100)
    new_per_batch = batch_size - old_per_batch
    steps = total // batch_size
    new_count = total - replay_count
    if not 0 < old_per_batch < batch_size:
        raise ValueError("Stratified replay requires both old and new entries per batch")
    if steps * old_per_batch > replay_count or steps * new_per_batch > new_count:
        raise ValueError("Buffer proportions cannot supply exact batch quotas without replacement")
    new_order = torch.randperm(new_count, generator=generator)
    old_order = torch.randperm(replay_count, generator=generator) + new_count
    for step in range(steps):
        indices = torch.cat((
            new_order[step * new_per_batch:(step + 1) * new_per_batch],
            old_order[step * old_per_batch:(step + 1) * old_per_batch],
        ))
        yield indices[torch.randperm(batch_size, generator=generator)]


def quantize_features_int8(features):
    """Symmetrically quantize an NxC feature tensor using one scale per channel."""
    if features.ndim != 2:
        raise ValueError("features must have shape NxC")
    features_float = features.float()
    scales = features_float.abs().amax(dim=0).clamp_min(1e-8) / 127.0
    quantized = torch.round(features_float / scales).clamp(-127, 127).to(torch.int8)
    return quantized, scales


def dequantize_features_int8(quantized, scales, dtype=torch.float16):
    """Restore per-channel symmetric int8 features to a floating dtype."""
    if quantized.ndim != 2 or scales.ndim != 1:
        raise ValueError("Expected quantized NxC features and C scales")
    if quantized.shape[1] != len(scales):
        raise ValueError("Feature dimension and scale count do not match")
    return (quantized.float() * scales.float()).to(dtype)
