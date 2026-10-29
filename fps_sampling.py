"""Exact greedy FPS on filtered predicted XYZ coordinates, without subsampling."""
import logging
import time

import torch


@torch.no_grad()
def farthest_point_sample(scene_coords, sample_count, generator, device=None):
    """Return unique CPU indices; squared Euclidean distance in 3D, seeded start.

    Every candidate participates. Complexity is O(N*K), memory O(N+K).
    Coincident points remain distinct candidate tuples; already selected indices
    are excluded, so even identical XYZs cannot cause repeated indices.
    """
    if scene_coords.ndim != 2 or scene_coords.shape[1] != 3:
        raise ValueError('scene_coords must have shape Nx3')
    n = len(scene_coords)
    if not 1 <= sample_count <= n:
        raise ValueError('sample_count must be between 1 and the candidate count')
    if not torch.isfinite(scene_coords).all():
        raise ValueError('FPS requires finite coordinates')
    device = torch.device(device) if device is not None else scene_coords.device
    # Use direct coordinate differences rather than the cancellation-prone
    # ||x||^2 + ||y||^2 - 2*x*y formula.
    xyz = scene_coords.detach().to(device=device, dtype=torch.float32).contiguous()
    nearest = torch.full((n,), float('inf'), device=device)
    selected_mask = torch.zeros(n, dtype=torch.bool, device=device)
    selected = torch.empty(sample_count, dtype=torch.long, device=device)
    current = torch.randint(n, (1,), generator=generator, device='cpu').to(device).squeeze(0)
    logger = logging.getLogger(__name__)
    started = time.perf_counter()
    logger.info('Exact XYZ FPS: %d candidates -> %d tuples on %s (no candidate cap)',
                n, sample_count, device)
    for step in range(sample_count):
        selected[step] = current
        selected_mask[current] = True
        if step + 1 < sample_count:
            distance = (xyz - xyz[current]).square().sum(dim=1)
            torch.minimum(nearest, distance, out=nearest)
            nearest.masked_fill_(selected_mask, -1.0)
            current = torch.argmax(nearest)
        if (step + 1) % 1000 == 0 or step + 1 == sample_count:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            logger.info('FPS selected %d/%d in %.1fs', step + 1, sample_count,
                        time.perf_counter() - started)
    return selected.cpu()
