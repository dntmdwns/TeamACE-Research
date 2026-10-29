# TeamACE: Continual Learning for Visual Localization

A curated research source snapshot exploring how an ACE visual-localization model can learn scenes sequentially while retaining performance on earlier scenes.

**Research contact:** [Seungjun Woo](https://github.com/dntmdwns), Stony Brook University.
This is collaborative team work; this repository does not claim sole authorship of the baseline or all extensions. See [ATTRIBUTION.md](ATTRIBUTION.md).

## Research questions

- How much does a single scene-coordinate regression head forget when trained on a new scene?
- Can replaying stored encoder features retain earlier-scene performance without storing source RGB images in the replay memory?
- How do random, coverage-based, error-based, and farthest-point feature selection compare under a fixed memory budget?
- What are the memory trade-offs of FP32, FP16, and INT8 feature storage?

The code includes experimental selection variants. Feature selection and preliminary feature-quantization experiments address separate questions; quantization is not presented as a verified explanation for the selection results.

## Code guide

| File | Purpose |
| --- | --- |
| `ace_trainer.py` | ACE training with replay and feature-selection extensions |
| `replay_buffer.py` | Replay sampling and feature encoding utilities |
| `fps_sampling.py` | Farthest-point sampling utilities |
| `precision.py` | Precision-related utilities |
| `train_ace.py`, `test_ace.py` | Training and localization evaluation entry points |
| `research/run_standard.sh` | Configurable sequential-learning experiment runner |
| `research/standard_runner.py` | Experiment setup, aggregation, and provenance helpers |
| `research/run_selection_experiments.py` | Historical Chess-to-Fire selection-study orchestration |
| `research/run_scene_order_selection.py` | Historical reverse-order selection-study orchestration |
| `research/test_*.py` | Focused checks for sampling and experiment orchestration |

## Setup and scope

This is a source-code portfolio, not a packaged pretrained model or a complete experiment archive. GPU training has not been rerun from this exported snapshot.

1. Follow the [original ACE setup instructions](https://github.com/nianticlabs/ace#installation). The included `environment.yml` records the upstream Linux/CUDA environment; platform-specific adjustments may be needed. The standard shell runner targets Linux with Bash 4+ and GNU coreutils.
2. Build the DSAC* extension using the upstream instructions and the included `dsacstar/` sources.
3. Obtain the ACE pretrained encoder from the [upstream ACE repository](https://github.com/nianticlabs/ace) and place `ace_encoder_pretrained.pt` in the repository root. Model weights are not redistributed here.
4. Obtain and prepare datasets using their original sources and terms; preparation scripts are provided under `datasets/`. Dataset files are not included.
5. Run `python train_ace.py --help`, `python test_ace.py --help`, or `bash research/run_standard.sh --help` in the configured environment to inspect available options.

Historical runners use different defaults from the current standard runner. Always specify and record the pose convention, scene order, training budget, precision, replay ratio, memory size, and seeds before comparing results. Calibrated-pose and pseudo-ground-truth-pose experiments must not be pooled as directly comparable results.

This export intentionally omits private operational notes, machine configuration, automated workflows, raw logs, datasets, model weights, and internal experiment archives. It starts with a clean Git history and is maintained separately from the working team repository.

## Attribution and license

Built on **Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses**, Eric Brachmann, Tommaso Cavallari, and Victor Adrian Prisacariu, CVPR 2023.

- [Original repository](https://github.com/nianticlabs/ace)
- [Paper](https://arxiv.org/abs/2305.14059)
- [Project page](https://nianticlabs.github.io/ace/)

**Copyright © Niantic, Inc. 2023. All rights reserved.**

The original [LICENSE](LICENSE) is preserved without modification. ACE-derived code is provided under its non-commercial terms; the dataset and DSAC* portions retain their applicable notices. Do not assume an MIT or unrestricted commercial license. Modified and added files are documented in [ATTRIBUTION.md](ATTRIBUTION.md).
