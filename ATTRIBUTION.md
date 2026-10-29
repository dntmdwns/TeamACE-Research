# Attribution and modification notices

Copyright © Niantic, Inc. 2023. All rights reserved.

This repository is based on the ACE source release and subsequent collaborative TeamACE research. The baseline and DSAC* components retain the notices in `LICENSE` and individual source files.

The working repository's recorded contributors include Eric Brachmann, Tommaso Cavallari, Zoë McGinnis, Seungjun Woo, jiinx_h, and synahn. Names are credited from recorded Git authorship; this is not a statement that all contributors participated in every extension. The private development history is not redistributed.

## Research modifications

Compared with the initially imported source snapshot, the following distributed source files have been modified:

- `LICENSE`
- `ace_network.py`
- `ace_trainer.py`
- `dataset.py`
- `test_ace.py`
- `train_ace.py`

The following distributed research/helper files were added after that imported snapshot:

- `fps_sampling.py`
- `precision.py`
- `replay_buffer.py`
- `research/compare_replay_buffers.py`
- `research/run_matched_replay_condition.py`
- `research/run_scene_order_selection.py`
- `research/run_selection_experiments.py`
- `research/run_standard.sh`
- `research/run_two_scene_sequential.py`
- `research/standard_runner.py`
- `research/summarize_replay_multiseed.py`
- `research/test_feature_dedup.py`
- `research/test_fps_sampling.py`
- `research/test_replay_buffer.py`
- `research/test_scene_offsets.py`
- `research/test_scene_order_selection.py`
- `research/test_selection_runner.py`
- `research/test_standard_runner.py`
- `research/verify_scene_order_results.py`

## Public-export changes

- Replaced the root README with a research overview and code guide.
- Replaced `.gitignore` to exclude generated outputs, model weights, local credentials, and runner files.
- Removed a machine-specific fallback checkout path from `research/run_scene_order_selection.py`; it resolves its checkout relative to the script.
- Generalized an internal-result-path comment in `research/run_standard.sh`.
- Omitted internal guides, workflows, logs, results archives, datasets, and model weights.
- Preserved the original license text verbatim; no additional permissive license is asserted.

The export documents experimental code rather than assigning ownership of all changes to the portfolio contact. Please cite the original ACE work when using the baseline.
