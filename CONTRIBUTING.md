# Research and contribution notes

This public snapshot is separate from the working team's infrastructure.

- Keep datasets, pretrained or trained model files, replay stores, raw logs, credentials, and machine-specific paths out of Git.
- The standard runner defaults to pseudo-ground-truth poses, a 4M training buffer per stage, legacy precision, a 10% stratified replay ratio, a 5-pixel reprojection filter, a deduplicated candidate pool, and seeds 2089-2091. Pass explicit arguments when changing the protocol.
- Distinguish `filtered_random` from unfiltered `random`.
- Record scene order, pose convention, training budget, precision, memory size, and seeds with results. Do not pool incompatible protocols.
- Do not claim a method is generally superior from one scene pair or one seed.
- Preserve upstream copyright and license notices, and document modifications in `ATTRIBUTION.md`.

The public repository has no automated workflows or attached self-hosted runners.
