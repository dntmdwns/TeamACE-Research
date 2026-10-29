#!/usr/bin/env python3
# HISTORICAL runner, kept to document past results; new experiments use research/run_standard.sh.
# Defaults to: calibrated (D-SLAM) poses (datasets/7scenes_*), Chess -> Fire without replay, legacy precision
# (train_ace.py default), 4M buffer, seed 2089.
"""Run the two-scene, one-head ACE sequential fine-tuning baseline."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


THRESHOLDS = ("10cm_5deg", "5cm_5deg", "2cm_2deg", "1cm_1deg")


def run(command, env):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, env=env)


def load_json(path):
    with path.open() as input_file:
        return json.load(input_file)


def command_output(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_summary(scene_a, scene_b, train_a, train_b, evaluations):
    a_before = evaluations["stage_a"]["scene_a"]
    a_after = evaluations["stage_b"]["scene_a"]
    forgetting = {
        threshold: a_before["accuracy_percent"][threshold] - a_after["accuracy_percent"][threshold]
        for threshold in THRESHOLDS
    }

    return {
        "method": "sequential_finetuning_no_replay",
        "scene_order": [scene_a, scene_b],
        "replay_bytes": 0,
        "training": {"stage_a": train_a, "stage_b": train_b},
        "evaluation": evaluations,
        "forgetting_scene_a_accuracy_points": forgetting,
        "scene_a_median_error_increase": {
            "translation_cm": (
                a_after["median_translation_error_cm"] - a_before["median_translation_error_cm"]
            ),
            "rotation_deg": (
                a_after["median_rotation_error_deg"] - a_before["median_rotation_error_deg"]
            ),
        },
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train one ACE head on scene A, continue it on scene B, and score both scenes."
    )
    parser.add_argument("--datasets_root", type=Path, default=repo_root / "datasets")
    parser.add_argument("--scene_a", default="7scenes_chess")
    parser.add_argument("--scene_b", default="7scenes_fire")
    parser.add_argument("--output_dir", type=Path,
                        default=repo_root / "research" / "results" / "two-scene-chess-fire-no-replay")
    parser.add_argument("--training_buffer_size", type=int, default=4_000_000)
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--gpu", default=None,
                        help="physical GPU index; sets CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--track_epochs", action="store_true",
                        help="checkpoint stage B after every epoch and evaluate a fixed diagnostic subset")
    parser.add_argument("--curve_test_images", type=int, default=200,
                        help="number of evenly spaced test images per scene for epoch curves")
    args = parser.parse_args()

    scene_a_path = args.datasets_root / args.scene_a
    scene_b_path = args.datasets_root / args.scene_b
    for scene_path in (scene_a_path, scene_b_path):
        if not (scene_path / "train").is_dir() or not (scene_path / "test").is_dir():
            parser.error(f"Expected train/ and test/ under {scene_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_a_head = args.output_dir / "stage_a_head.pt"
    stage_b_head = args.output_dir / "stage_b_head.pt"
    train_a_json = args.output_dir / "train_stage_a.json"
    train_b_json = args.output_dir / "train_stage_b.json"
    checkpoint_dir = args.output_dir / "stage_b_checkpoints"

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    common_train = [
        "--training_buffer_size", args.training_buffer_size,
        "--batch_size", args.batch_size,
        "--epochs", args.epochs,
        "--seed", args.seed,
    ]

    source_files = [
        repo_root / "train_ace.py",
        repo_root / "ace_trainer.py",
        repo_root / "test_ace.py",
        Path(__file__).resolve(),
    ]
    gpu_model = None
    if args.gpu is not None:
        gpu_model = command_output([
            "nvidia-smi", f"--id={args.gpu}", "--query-gpu=name", "--format=csv,noheader",
        ])

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "method": "sequential_finetuning_no_replay",
        "scene_order": [args.scene_a, args.scene_b],
        "pose_source": "calibrated 7-Scenes ground-truth poses",
        "seed": args.seed,
        "training_buffer_size": args.training_buffer_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "gpu": args.gpu,
        "gpu_model": gpu_model,
        "ace_base_commit": command_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "experiment_source_sha256": source_fingerprint(source_files),
        "coordinate_system": "native calibrated 7-Scenes coordinates; no cross-scene alignment",
        "replay_bytes": 0,
        "track_epochs": args.track_epochs,
        "curve_test_images": args.curve_test_images if args.track_epochs else None,
    }
    with (args.output_dir / "experiment.json").open("w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    run([
        args.python, repo_root / "train_ace.py", scene_a_path, stage_a_head,
        *common_train, "--training_metrics_json", train_a_json,
    ], env)

    evaluations = {"stage_a": {}, "stage_b": {}}
    for scene_key, scene_path in (("scene_a", scene_a_path), ("scene_b", scene_b_path)):
        metrics_path = args.output_dir / f"eval_stage_a_{scene_key}.json"
        run([
            args.python, repo_root / "test_ace.py", scene_path, stage_a_head,
            "--session", f"stage_a_{scene_key}", "--metrics_json", metrics_path,
        ], env)
        evaluations["stage_a"][scene_key] = load_json(metrics_path)

    stage_b_train_command = [
        args.python, repo_root / "train_ace.py", scene_b_path, stage_b_head,
        *common_train, "--resume_head", stage_a_head,
        "--training_metrics_json", train_b_json,
    ]
    if args.track_epochs:
        stage_b_train_command.extend([
            "--checkpoint_dir", checkpoint_dir,
            "--checkpoint_every", 1,
        ])
    run(stage_b_train_command, env)

    if args.track_epochs:
        curve = {
            "diagnostic_subset_images_per_scene": args.curve_test_images,
            "note": "Evenly spaced fixed subsets; use full-test metrics in summary.json for final reporting.",
            "epochs": [],
        }
        epoch_heads = [(0, stage_a_head)] + [
            (epoch, checkpoint_dir / f"epoch_{epoch:03d}.pt")
            for epoch in range(1, args.epochs + 1)
        ]
        curve_dir = args.output_dir / "epoch_metrics"
        for epoch, epoch_head in epoch_heads:
            epoch_entry = {"epoch": epoch}
            for scene_key, scene_path in (("scene_a", scene_a_path), ("scene_b", scene_b_path)):
                metrics_path = curve_dir / f"epoch_{epoch:03d}_{scene_key}.json"
                run([
                    args.python, repo_root / "test_ace.py", scene_path, epoch_head,
                    "--session", f"curve_epoch_{epoch:03d}_{scene_key}",
                    "--max_test_images", args.curve_test_images,
                    "--metrics_json", metrics_path,
                ], env)
                epoch_entry[scene_key] = load_json(metrics_path)
            curve["epochs"].append(epoch_entry)
        with (args.output_dir / "learning_curve.json").open("w") as curve_file:
            json.dump(curve, curve_file, indent=2)
            curve_file.write("\n")

    for scene_key, scene_path in (("scene_a", scene_a_path), ("scene_b", scene_b_path)):
        metrics_path = args.output_dir / f"eval_stage_b_{scene_key}.json"
        run([
            args.python, repo_root / "test_ace.py", scene_path, stage_b_head,
            "--session", f"stage_b_{scene_key}", "--metrics_json", metrics_path,
        ], env)
        evaluations["stage_b"][scene_key] = load_json(metrics_path)

    summary = build_summary(
        args.scene_a,
        args.scene_b,
        load_json(train_a_json),
        load_json(train_b_json),
        evaluations,
    )
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    print(f"Saved final score summary to: {summary_path}")


if __name__ == "__main__":
    main()
