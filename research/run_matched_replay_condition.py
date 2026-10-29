#!/usr/bin/env python3
# HISTORICAL runner, kept to document past results; new experiments use research/run_standard.sh.
# Defaults to: calibrated (D-SLAM) poses (datasets/7scenes_*), legacy precision (train_ace.py default), random
# (unfiltered) or voxel selection (required), 200k stored features, 10% shuffle replay, 4M buffer, seed 2089.
"""Evaluate one replay condition from a fixed, already-trained Chess head."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command, env):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, env=env)


def load_json(path):
    with path.open() as input_file:
        return json.load(input_file)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_a_head", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--datasets_root", type=Path, default=repo_root / "datasets")
    parser.add_argument("--selection", choices=["random", "voxel"], required=True)
    parser.add_argument("--feature_encoding", choices=["fp16", "int8"], required=True)
    parser.add_argument("--replay_buffer_features", type=int, default=200000)
    parser.add_argument("--replay_ratio_percent", type=float, default=10.0)
    parser.add_argument("--replay_voxel_size_m", type=float, default=0.25)
    parser.add_argument("--replay_reprojection_error_px", type=float, default=5.0)
    parser.add_argument("--training_buffer_size", type=int, default=4_000_000)
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not args.stage_a_head.is_file():
        parser.error(f"Stage-A head not found: {args.stage_a_head}")
    if not 0 < args.replay_ratio_percent < 100:
        parser.error("replay_ratio_percent must be in (0, 100)")

    chess = args.datasets_root / "7scenes_chess"
    fire = args.datasets_root / "7scenes_fire"
    for scene in (chess, fire):
        if not (scene / "train").is_dir() or not (scene / "test").is_dir():
            parser.error(f"Expected train/ and test/ under {scene}")

    args.output_dir.mkdir(parents=True)
    replay_buffer = args.output_dir / "replay_buffer.pt"
    export_metrics = args.output_dir / "export.json"
    stage_b_head = args.output_dir / "stage_b_head.pt"
    train_metrics = args.output_dir / "train_stage_b.json"

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
    run([
        args.python, repo_root / "train_ace.py", chess, args.output_dir / "unused_head.pt",
        *common_train,
        "--resume_head", args.stage_a_head,
        "--export_only",
        "--export_replay_buffer", replay_buffer,
        "--replay_buffer_features", args.replay_buffer_features,
        "--replay_selection", args.selection,
        "--replay_feature_encoding", args.feature_encoding,
        "--replay_voxel_size_m", args.replay_voxel_size_m,
        "--replay_reprojection_error_px", args.replay_reprojection_error_px,
        "--training_metrics_json", export_metrics,
    ], env)

    run([
        args.python, repo_root / "train_ace.py", fire, stage_b_head,
        *common_train,
        "--resume_head", args.stage_a_head,
        "--replay_buffer", replay_buffer,
        "--replay_ratio_percent", args.replay_ratio_percent,
        "--training_metrics_json", train_metrics,
    ], env)

    evaluation = {}
    session_prefix = args.output_dir.name.replace("_", "-")
    for scene_key, scene in (("chess", chess), ("fire", fire)):
        metrics_path = args.output_dir / f"eval_full_{scene_key}.json"
        run([
            args.python, repo_root / "test_ace.py", scene, stage_b_head,
            "--session", f"{session_prefix}-{scene_key}",
            "--metrics_json", metrics_path,
        ], env)
        evaluation[scene_key] = load_json(metrics_path)

    chess_accuracy = evaluation["chess"]["accuracy_percent"]["5cm_5deg"]
    fire_accuracy = evaluation["fire"]["accuracy_percent"]["5cm_5deg"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_a_head": str(args.stage_a_head),
        "seed": args.seed,
        "selection": args.selection,
        "feature_encoding": args.feature_encoding,
        "replay_features": args.replay_buffer_features,
        "replay_ratio_percent": args.replay_ratio_percent,
        "replay_buffer_bytes": replay_buffer.stat().st_size,
        "training_buffer_size": args.training_buffer_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "voxel_size_m": args.replay_voxel_size_m if args.selection == "voxel" else None,
        "reprojection_error_px_l1": (
            args.replay_reprojection_error_px if args.selection == "voxel" else None
        ),
        "chess_5cm_5deg_percent": chess_accuracy,
        "fire_5cm_5deg_percent": fire_accuracy,
        "average_5cm_5deg_percent": (chess_accuracy + fire_accuracy) / 2,
        "export": load_json(export_metrics),
        "training": load_json(train_metrics),
        "evaluation": evaluation,
    }
    with (args.output_dir / "summary.json").open("w") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(f"Saved matched replay result to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
