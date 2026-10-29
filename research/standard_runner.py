#!/usr/bin/env python3
"""Helpers for research/run_standard.sh: prerequisite checks, environment capture, stage-A head records, the SETUP
block and summaries. Run with the experiment's Python interpreter; only `env`, `run-summary` and `summarize` (for
replay stores) import torch.

usage: standard_runner.py prereqs ... | env --gpu N | write-settings OUT k=v ... | head-record ... |
       head-check ... | setup OUT | run-summary RUN_DIR ... | summarize OUT |
       scene-center --dataset DIR [--offset=X,Y,Z] | midpoint --a=X,Y,Z --b=X,Y,Z
"""
import argparse
import hashlib
import json
import platform
import shutil
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KEYS = ("10cm_5deg", "5cm_5deg", "2cm_2deg", "1cm_1deg")
# Settings a stage-A head depends on. A reused head's record must match the current run on every one of them.
# The code is compared by the sha256 of the training files, not by commit, so doc-only commits keep heads valid;
# the commit is still recorded.
HEAD_SETTING_KEYS = ("scene", "pose_version", "training_buffer_size", "batch_size", "epochs", "precision", "seed",
                     "scene_offset", "head_center", "fp32_coordinates", "training_code_sha256", "gpu_name",
                     "environment")
# Head settings added after the first records were written; a record without one used the default.
HEAD_SETTING_DEFAULTS = {"scene_offset": "", "head_center": "scene_a", "fp32_coordinates": "0"}
TRAINING_FILES = ("ace_trainer.py", "ace_network.py", "dataset.py", "train_ace.py", "ace_loss.py", "ace_util.py",
                  "precision.py", "replay_buffer.py")
# The team standard (CONTRIBUTING.md); any other value is listed as a deviation in SETUP.md.
STANDARD = {"pose_version": "pgt", "training_buffer_size": "4000000", "batch_size": "5120", "epochs": "16",
            "precision": "legacy", "feature_encoding": "fp16", "replay_ratio_percent": "10",
            "batch_mode": "stratified", "reprojection_filter_px": "5", "drop_repeated_candidates": "1",
            "seeds": "2089,2090,2091", "export_threads": "1", "offset_a": "", "offset_b": "",
            "head_center": "scene_a", "fp32_coordinates": "0"}
POSE_DIR_PREFIX = {"pgt": "pgt_7scenes_", "calibrated": "7scenes_"}
CALIBRATED_FOCAL = 525.0


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command):
    return subprocess.run(command, capture_output=True, text=True)


# ---------------------------------------------------------------------------------------------------- prerequisites

def check_dataset(path, pose_version):
    """Problems with one scene folder (empty list if fine). pgt calibration files hold per-frame SfM focal lengths;
    calibrated (D-SLAM) ones all hold 525."""
    if not path.is_dir():
        return [f"{path}: missing (set up with `cd datasets && ./setup_7scenes.py --poses {pose_version}`, "
                f"or pass --datasets_dir)"]
    problems = []
    for split in ("train", "test"):
        counts = {}
        for sub in ("rgb", "poses", "calibration"):
            folder = path / split / sub
            counts[sub] = sum(1 for _ in folder.iterdir()) if folder.is_dir() else 0
        if 0 in counts.values() or len(set(counts.values())) != 1:
            problems.append(f"{path}/{split}: incomplete (rgb/poses/calibration files: "
                            f"{counts['rgb']}/{counts['poses']}/{counts['calibration']})")
            continue
        focals = {float(f.read_text().split()[0]) for f in (path / split / "calibration").iterdir()}
        looks_pgt = focals != {CALIBRATED_FOCAL}
        if looks_pgt != (pose_version == "pgt"):
            found = "pgt (per-frame focal lengths)" if looks_pgt else "calibrated (all focal lengths 525)"
            problems.append(f"{path}/{split}: poses look {found}, not {pose_version}")
    return problems


def gpu_table():
    """{index: (name, uuid, memory_used_mib)} and the set of GPU UUIDs running compute processes."""
    gpus = run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader,nounits"])
    apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    if gpus.returncode != 0:
        raise RuntimeError(gpus.stderr.strip() or "nvidia-smi failed")
    table = {}
    for line in gpus.stdout.strip().splitlines():
        index, name, uuid, used = (part.strip() for part in line.split(","))
        table[index] = (name, uuid, int(used))
    busy = {line.split(",")[0].strip() for line in apps.stdout.strip().splitlines() if line.strip()}
    return table, busy


def cmd_prereqs(a):
    missing = []
    for scene in a.scenes.split(","):
        missing += check_dataset(Path(a.datasets_dir) / f"{POSE_DIR_PREFIX[a.pose_version]}{scene}", a.pose_version)
    if not Path(a.encoder).is_file():
        missing.append(f"encoder weights missing: {a.encoder}")
    probe = run([a.python, "-c", "import torch, cv2, numpy, dsacstar; "
                 "assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'"])
    if probe.returncode != 0:
        last = (probe.stderr.strip().splitlines() or ["unknown error"])[-1]
        missing.append(f"{a.python} cannot import torch/cv2/numpy/dsacstar or has no CUDA: {last} "
                       f"(build DSAC* with `cd dsacstar && python setup.py install` in that environment)")
    for tool in ("tmux", "nvidia-smi", "git", "sha256sum"):
        if shutil.which(tool) is None:
            missing.append(f"`{tool}` not found on PATH")
    if shutil.which("nvidia-smi"):
        try:
            table, busy = gpu_table()
        except RuntimeError as error:
            missing.append(f"nvidia-smi: {error}")
        else:
            for gpu in a.gpus.split(","):
                if gpu not in table:
                    missing.append(f"GPU {gpu} not found (available: {', '.join(sorted(table)) or 'none'})")
                    continue
                name, uuid, used = table[gpu]
                if uuid in busy or used > a.max_gpu_mem_used_mib:
                    missing.append(f"GPU {gpu} ({name}) is in use: {used} MiB used"
                                   f"{', compute processes running' if uuid in busy else ''} "
                                   f"(limit --max_gpu_mem_used_mib {a.max_gpu_mem_used_mib})")
    if missing:
        print("Prerequisites missing:")
        print("\n".join(f"  - {item}" for item in missing))
        return 1
    print("Prerequisites OK: datasets with "
          f"{a.pose_version} poses, DSAC* importable, tmux available, GPU(s) {a.gpus} free.")
    return 0


# ------------------------------------------------------------------------------------------------------ environment

def environment(gpu):
    """GPU model and software versions of this process's environment (results are comparable only within one)."""
    import cv2
    import numpy
    import torch
    driver = run(["nvidia-smi", f"--id={gpu}", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    name, _, driver_version = driver.stdout.strip().partition(", ")
    return {
        "gpu_name": name or "unknown",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
            "nvidia_driver": driver_version or "unknown",
        },
    }


def cmd_env(a):
    env = environment(a.gpu)
    if a.provenance:   # the provenance.txt lines
        import torch
        print(f"torch: {torch.__version__} cuda {torch.version.cuda}")
        print(f"environment: {json.dumps(env)}")
    else:
        print(json.dumps(env, indent=2))
    return 0


# ---------------------------------------------------------------------------------------------- settings and heads

def cmd_write_settings(a):
    settings = dict(item.split("=", 1) for item in a.items)
    Path(a.path).write_text(json.dumps(settings, indent=2) + "\n")
    return 0


def head_settings(settings, seed, env):
    """What the current run requires of seed's stage-A head."""
    return {"scene": settings["scenes"].split(",")[0], "pose_version": settings["pose_version"],
            "training_buffer_size": int(settings["training_buffer_size"]), "batch_size": int(settings["batch_size"]),
            "epochs": int(settings["epochs"]), "precision": settings["precision"], "seed": int(seed),
            "scene_offset": settings.get("offset_a", ""),
            "head_center": settings.get("head_center_xyz") or "scene_a",
            "fp32_coordinates": settings.get("fp32_coordinates", "0"),
            "training_code_sha256": {name: sha256(Path(settings["repo"]) / name) for name in TRAINING_FILES},
            "gpu_name": env["gpu_name"], "environment": env["environment"]}


def head_mismatches(recorded, current):
    """Human-readable differences between a head's recorded settings and the current run's."""
    out = []
    for key in HEAD_SETTING_KEYS:
        old, new = recorded.get(key, HEAD_SETTING_DEFAULTS.get(key, "<not recorded>")), current[key]
        if isinstance(new, dict) and isinstance(old, dict):
            out += [f"{key}.{k}: head {old.get(k, '<not recorded>')!r}, this run {v!r}"
                    for k, v in new.items() if old.get(k, "<not recorded>") != v]
        elif old != new:
            out.append(f"{key}: head {old!r}, this run {new!r}")
    return out


def sidecar(head):
    return Path(head).with_suffix(".json")


def cmd_head_record(a):
    """Write the JSON record next to a stage-A head the runner just trained."""
    settings = json.loads(Path(a.settings).read_text())
    record = {"settings": head_settings(settings, a.seed, environment(a.gpu)),
              "head_sha256": sha256(a.head), "head_file": Path(a.head).name, "dataset": a.dataset,
              "host": socket.gethostname(), "gpu_index": a.gpu, "git_commit": settings["git_commit"],
              "git_branch": settings["git_branch"],
              "runner_sha256": settings["runner_sha256"], "created_utc": now(), "train_command": a.command}
    sidecar(a.head).write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {sidecar(a.head)}")
    return 0


def cmd_head_check(a):
    """Compare a reused head's record with the current run; write the verdict to --out. Exit 1 if refused."""
    settings = json.loads(Path(a.settings).read_text())
    current = head_settings(settings, a.seed, environment(a.gpu))
    head, record_path = Path(a.head), sidecar(a.head)
    verdict = {"seed": int(a.seed), "head": str(head.resolve()), "record": str(record_path.resolve()),
               "override": a.allow_mismatch, "mismatches": [], "problems": []}
    if not head.is_file():
        verdict["problems"].append(f"head file missing: {head}")
    expected = a.expected_sha256 or None
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        verdict["mismatches"] = head_mismatches(record.get("settings", {}), current)
        if expected and expected != record.get("head_sha256"):
            verdict["problems"].append("--stage_a_head_sha256 disagrees with the record's head_sha256")
        expected = record.get("head_sha256")
    else:
        verdict["mismatches"].append(f"record missing: {record_path}")
    if head.is_file():
        verdict["head_sha256"] = sha256(head)
        if expected is None:
            verdict["problems"].append("no sha256 to check the head against (no record; pass --stage_a_head_sha256)")
        elif verdict["head_sha256"] != expected:
            verdict["problems"].append(f"head sha256 {verdict['head_sha256']} != expected {expected}")
    refused = bool(verdict["problems"]) or (bool(verdict["mismatches"]) and not a.allow_mismatch)
    verdict["status"] = "refused" if refused else ("deviation (override)" if verdict["mismatches"] else "match")
    Path(a.out).write_text(json.dumps(verdict, indent=2) + "\n")
    for line in verdict["problems"] + verdict["mismatches"]:
        print(f"  seed {a.seed}: {line}")
    return 1 if refused else 0


# ------------------------------------------------------------------------------------------------------------ SETUP

def coordinate_frame(s):
    """One line for SETUP.md: scene offsets, head center and coordinate precision."""
    a_scene, b_scene = s["scenes"].split(",")
    offsets = (f"offsets {a_scene} {s.get('offset_a') or 'none'}, {b_scene} {s.get('offset_b') or 'none'} m "
               "(camera positions shifted, rotations unchanged)")
    center = {"scene_a": f"{a_scene}'s mean camera position (ACE default)",
              "midpoint": f"midpoint of both scenes' mean camera positions ({s.get('head_center_xyz')})"}.get(
        s.get("head_center", "scene_a"), f"explicit {s.get('head_center')}")
    precision = ("fp32 final layer and center addition (no autocast, no TF32), fp32 heads"
                 if s.get("fp32_coordinates") == "1" else "final layer under autocast (fp16), as ACE")
    return (f"{offsets}; head center: {center}; coordinates: {precision}; region centers "
            f"{a_scene} {s.get('center_a')}, {b_scene} {s.get('center_b')}")


def cmd_setup(a):
    """SETUP.md for an output folder: the CONTRIBUTING.md SETUP block plus every deviation from the standard."""
    out = Path(a.out)
    s = json.loads((out / "settings.json").read_text())
    env = json.loads((out / "environment.json").read_text())
    a_scene, b_scene = s["scenes"].split(",")
    seeds, gpus = s["seeds"].split(","), s["gpus"].split(",")
    queue = "; ".join(f"GPU {g}: {', '.join(seeds[i::len(gpus)])}" for i, g in enumerate(gpus) if seeds[i::len(gpus)])
    dedup = "on (`--replay_drop_repeated_candidates`)" if s["drop_repeated_candidates"] == "1" else "off"
    heads = (f"reused, `{s['stage_a_head']}`" if s["stage_a_head"]
             else f"trained per seed as stage A (`stage_a/seed<N>/stage_a_head.pt` + `.json`)")
    rows = [
        ("Pose", f"{s['pose_version']} (`{s['datasets_dir']}/{POSE_DIR_PREFIX[s['pose_version']]}<scene>`)"),
        ("Scenes", f"{a_scene} → {b_scene}"),
        ("Buffer", f"{int(s['training_buffer_size']):,} per stage, batch {s['batch_size']}, {s['epochs']} epochs"),
        ("Precision", f"{s['precision']} (`--precision {s['precision']}`), replay features {s['feature_encoding']}"),
        ("Batch mode", s["batch_mode"]),
        ("Replay ratio", f"{s['replay_ratio_percent']}% (0 for `none`)"),
        ("Methods, K", f"{s['methods']}; K = {s['ks']}" + (f"; voxel_weighted alpha = {s.get('voxel_alphas', '1')}"
                                                          if "voxel_weighted" in s["methods"].split(",") else "")),
        ("Filter", f"{s['reprojection_filter_px']} px L1 reprojection filter (filtered methods)"),
        ("Dedup", dedup),
        ("Seeds", ", ".join(seeds)),
        ("GPU", f"{socket.gethostname()}, {env['gpu_name']}; {queue}"),
        ("Environment", ", ".join(f"{k} {v}" for k, v in env["environment"].items())),
        ("Commit", f"`{s['git_commit'][:7]}` (branch `{s['git_branch']}`), runner sha256 `{s['runner_sha256'][:12]}`"),
        ("Stage-A heads", heads),
        ("Export", f"single-threaded CPU (`OMP_NUM_THREADS={s['export_threads']} MKL_NUM_THREADS={s['export_threads']} "
                   "OMP_DYNAMIC=FALSE`)" if s["export_threads"] == "1" else f"{s['export_threads']} CPU threads"),
        ("Stages", "export only (no stage-B training or evaluation)" if s["export_only"] == "1" else "full"),
        ("Coordinate frame", coordinate_frame(s)),
    ]
    deviations = [f"`{k}` = {s.get(k, v) or 'off'} (standard: {v or 'off'})"
                  for k, v in STANDARD.items() if s.get(k, v) != v]
    for path in sorted(out.glob("stage_a/seed*/head_check.json")):
        check = json.loads(path.read_text())
        if check["mismatches"]:
            deviations.append(f"seed {check['seed']} stage-A head `{check['head']}` used with "
                              f"`--allow_head_mismatch`: " + "; ".join(check["mismatches"])
                              + (f". Reason: {s['head_mismatch_reason']}" if s.get("head_mismatch_reason") else ""))
    lines = ["## SETUP", "", "| Setting | Value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "Deviations from the standard setup:", ""] + ([f"- {d}" for d in deviations] or ["- none"])
    lines += ["", f"Command: `{s['invocation']}`", ""]
    (out / "SETUP.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


# -------------------------------------------------------------------------------------------------------- summaries

def eval_row(path):
    m = json.loads(Path(path).read_text())
    row = {**{k: m["accuracy_percent"][k] for k in KEYS}, "median_translation_cm": m["median_translation_error_cm"],
           "median_rotation_deg": m["median_rotation_error_deg"], "num_frames": m["num_frames"]}
    if "nearest_region" in m:   # % of test frames whose estimated camera position is nearest to each scene's center
        row.update({f"region_{name}_percent": r["percent"] for name, r in m["nearest_region"].items()})
    return row


def regions(row):
    return ", ".join(f"{k[len('region_'):-len('_percent')]} {v:.1f}%" for k, v in row.items()
                     if k.startswith("region_")) or "—"


def table(rows):
    lines = ["| evaluation | 10cm/5° | 5cm/5° | 2cm/2° | 1cm/1° | median | frames | nearest region |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for label, r in rows.items():
        lines.append(f"| {label} | " + " | ".join(f"{r[k]:.1f}%" for k in KEYS)
                     + f" | {r['median_translation_cm']:.1f} cm, {r['median_rotation_deg']:.2f}° | {r['num_frames']} "
                     + f"| {regions(r)} |")
    return lines


def store_facts(run_dir, scene_a):
    import torch
    export = json.loads((run_dir / "export.json").read_text())
    replay_file = run_dir / f"replay_{scene_a}.pt"
    payload = torch.load(replay_file, map_location="cpu", weights_only=False)
    meta, data = payload["selection_metadata"], payload["data"]
    assert payload["num_features"] == export["replay_features_stored"]
    # Duplicates: stored entries identical in every field (raw bits), i.e. repeated copies of one (image, pixel).
    entries = torch.cat([data[k].flatten(1).view(torch.int16 if data[k].dtype == torch.float16 else torch.int32).long()
                         for k in ("features", "target_px", "gt_poses_inv", "intrinsics")], dim=1)
    return {"selection": payload["selection"], "reprojection_filter_px_l1": meta.get("reprojection_error_px_l1"),
            "repeated_candidates_dropped": meta.get("repeated_candidates_dropped"),
            "filtered_candidates": meta.get("valid_candidates"), "stored_k": payload["num_features"],
            "replay_store_bytes": replay_file.stat().st_size, "feature_encoding": payload["feature_encoding"],
            "duplicate_entries_in_store": len(entries) - len(torch.unique(entries, dim=0)),
            "selected_occupied_voxels": meta.get("selected_occupied_voxels"),
            "voxel_alpha": meta.get("voxel_alpha"),
            "selected_error_mean_px": meta.get("selected_error_mean_px"),
            "candidate_geometry_sha256": meta.get("candidate_geometry_sha256"),
            "replay_store_sha256": sha256(replay_file)}


def cmd_run_summary(a):
    """summary.json + summary.md for one run folder (kind: stage_a, replay or none)."""
    run_dir, stage_a = Path(a.run_dir), Path(a.stage_a_dir)
    scene_a, scene_b = a.scenes.split(",")
    result, lines = {}, []
    if a.kind == "replay":
        store = store_facts(run_dir, scene_a)
        result["replay_store"] = store
        fmt = lambda v, spec: "n/a" if v is None else format(v, spec)
        lines += [f"selection: {store['selection']} | filtered candidates: {fmt(store['filtered_candidates'], ',')} "
                  f"(after dropping {fmt(store['repeated_candidates_dropped'], ',')} repeated copies) | "
                  f"stored K: {store['stored_k']:,} | replay store: {store['replay_store_bytes']:,} bytes "
                  f"({store['feature_encoding']})"
                  + ("" if store["voxel_alpha"] is None else f" | voxel alpha: {store['voxel_alpha']:g}"),
                  f"pool sha256: {store['candidate_geometry_sha256']} | duplicates in store: "
                  f"{store['duplicate_entries_in_store']} | voxels covered: {store['selected_occupied_voxels']} | "
                  f"mean error: {fmt(store['selected_error_mean_px'], '.2f')} px"]
    train_file = run_dir / ("train_stage_a.json" if a.kind == "stage_a" else "train_stage_b.json")
    if train_file.is_file():
        t = json.loads(train_file.read_text())
        result["training"] = {k: t.get(k) for k in ("total_seconds", "head_training_seconds",
                                                     "peak_gpu_memory_allocated_bytes", "model_bytes",
                                                     "replay_ratio_percent", "replay_batch_mode", "replay_per_batch",
                                                     "replay_features_injected")}
        tr = result["training"]
        lines.append(f"training: {tr['total_seconds']:.0f} s, peak GPU memory {tr['peak_gpu_memory_allocated_bytes'] / 2**30:.2f} GiB, "
                     f"model {tr['model_bytes']:,} bytes | replay: {tr['replay_ratio_percent']:g}% "
                     f"{tr['replay_batch_mode']}, {tr['replay_per_batch'] or 'variable'} per batch, "
                     f"{tr['replay_features_injected']:,} entries")
    rows = {}
    if a.kind == "stage_a":
        for scene in (scene_a, scene_b):
            if (run_dir / f"eval_stage_a_{scene}.json").is_file():
                rows[f"{scene} after {scene_a}"] = eval_row(run_dir / f"eval_stage_a_{scene}.json")
    else:
        if (stage_a / f"eval_stage_a_{scene_a}.json").is_file():
            rows[f"{scene_a} before {scene_b} (stage A)"] = eval_row(stage_a / f"eval_stage_a_{scene_a}.json")
        for scene in (scene_a, scene_b):
            if (run_dir / f"eval_stage_b_{scene}.json").is_file():
                rows[f"{scene} after {scene_b}"] = eval_row(run_dir / f"eval_stage_b_{scene}.json")
    result["evaluation"] = rows
    if rows:
        lines += [""] + table(rows)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def mean_sd(values):
    if not values:
        return "—"
    return f"{statistics.mean(values):.1f}" + (f" ± {statistics.stdev(values):.1f}" if len(values) > 1 else "")


def method_label(folder):
    """Condition name of a method folder: voxel_weighted_alpha<a> -> voxel_weighted (alpha = a)."""
    if folder == "random":
        return "random (unfiltered)"
    if folder.startswith("voxel_weighted_alpha"):
        return f"voxel_weighted (alpha = {folder[len('voxel_weighted_alpha'):]})"
    return folder


def method_order(folder):
    """Sort method folders by name, voxel_weighted alphas numerically."""
    if folder.startswith("voxel_weighted_alpha"):
        return ("voxel_weighted", float(folder[len("voxel_weighted_alpha"):]))
    return (folder, 0.0)


def conditions(out):
    """(label, folder relative to out) for every condition, stage A first, then no replay, then method/K."""
    found = [("stage A only", "stage_a")]
    if (out / "none").is_dir():
        found.append(("no replay", "none"))
    method_dirs = [p for p in out.iterdir() if p.is_dir() and p.name not in ("stage_a", "none", "logs")]
    for method_dir in sorted(method_dirs, key=lambda p: method_order(p.name)):
        for k_dir in sorted(method_dir.glob("K*"), key=lambda p: int(p.name[1:])):
            name = method_label(method_dir.name)
            found.append((f"{name}, K = {int(k_dir.name[1:]):,}", f"{method_dir.name}/{k_dir.name}"))
    return found


def cmd_summarize(a):
    """summary.md, summary.csv and summary.json for an output folder: mean ± sample SD over finished seeds."""
    out = Path(a.out)
    s = json.loads((out / "settings.json").read_text())
    scene_a, scene_b = s["scenes"].split(",")
    seeds = s["seeds"].split(",")
    per_run, md = [], []
    md += [f"# Summary: {scene_a} → {scene_b}", "", f"Generated {now()} from the per-run `summary.json` files. "
           "Accuracy (%) after the last stage, mean ± sample SD over the seeds that finished (n).", ""]
    header = ("| Condition | Scene | n | 10cm/5° | 5cm/5° | 2cm/2° | 1cm/1° | Forgetting at 5cm/5° (points) "
              "| Nearest region (% of frames) | Replay store (bytes) | Pool sha256 (seed: first 12) | Stage time (s) "
              "| Peak GPU memory (GiB) |")
    md += [header, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |"]
    for label, folder in conditions(out):
        runs = {}
        for seed in seeds:
            path = out / folder / f"seed{seed}" / "summary.json"
            if path.is_file():
                runs[seed] = json.loads(path.read_text())
        if not runs:
            continue
        stores = {seed: r["replay_store"] for seed, r in runs.items() if "replay_store" in r}
        bytes_ = ", ".join(f"{b:,}" for b in sorted({st["replay_store_bytes"] for st in stores.values()})) or "—"
        pools = "; ".join(f"{seed}: {st['candidate_geometry_sha256'][:12]}" for seed, st in stores.items()
                          if st.get("candidate_geometry_sha256")) or "—"
        seconds = [r["training"]["total_seconds"] for r in runs.values() if "training" in r]
        memory = [r["training"]["peak_gpu_memory_allocated_bytes"] / 2**30 for r in runs.values() if "training" in r]
        for seed, r in runs.items():
            per_run.append({"condition": label, "folder": folder, "seed": int(seed), **r.get("replay_store", {}),
                            **{f"train_{k}": v for k, v in r.get("training", {}).items()},
                            **{f"{ev}|{k}": v for ev, row in r["evaluation"].items() for k, v in row.items()}})
        after = scene_a if folder == "stage_a" else scene_b
        scene_rows = [(scene, f"{scene} after {after}") for scene in (scene_a, scene_b)]
        if not any(lbl in r["evaluation"] for r in runs.values() for _, lbl in scene_rows):
            md.append(f"| {label} | (export only) | {len(runs)} | | | | | | | {bytes_} | {pools} | | |")
            continue
        for i, (scene, lbl) in enumerate(scene_rows):
            rows = [r["evaluation"][lbl] for r in runs.values() if lbl in r["evaluation"]]
            forgetting = "—"
            if folder != "stage_a" and scene == scene_a:
                before = f"{scene_a} before {scene_b} (stage A)"
                diffs = [r["evaluation"][before]["5cm_5deg"] - r["evaluation"][lbl]["5cm_5deg"]
                         for r in runs.values() if before in r["evaluation"] and lbl in r["evaluation"]]
                forgetting = mean_sd(diffs)
            first = i == 0
            region_keys = list(dict.fromkeys(k for r in rows for k in r if k.startswith("region_")))
            region_cell = ", ".join(f"{k[len('region_'):-len('_percent')]} {mean_sd([r[k] for r in rows if k in r])}"
                                    for k in region_keys) or "—"
            md.append(f"| {label if first else ''} | {scene} | {len(rows)} | "
                      + " | ".join(mean_sd([r[k] for r in rows]) for k in KEYS)
                      + f" | {forgetting} | {region_cell} | {bytes_ if first else ''} | {pools if first else ''} | "
                      + (f"{mean_sd(seconds)} | {max(memory):.2f}" if first and seconds else " | ") + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(per_run, indent=2) + "\n")
    columns = list(dict.fromkeys(k for row in per_run for k in row))
    csv_lines = [",".join(columns)] + [",".join("" if row.get(c) is None else str(row.get(c)) for c in columns)
                                       for row in per_run]
    (out / "summary.csv").write_text("\n".join(csv_lines) + "\n")
    print("\n".join(md))
    return 0


# ------------------------------------------------------------------------------------------------ coordinate frame

def parse_xyz(text):
    values = [float(v) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"expected x,y,z, got {text!r}")
    return values


def format_xyz(values):
    """float32 values as x,y,z with the shortest text that reads back as the same float32."""
    import numpy
    return ",".join(str(numpy.float32(v)) for v in values)


def cmd_scene_center(a):
    """Mean training camera position of a (shifted) scene, computed by the training code's own dataset class."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dataset import CamLocDataset
    offset = parse_xyz(a.offset) if a.offset else None
    print(format_xyz(CamLocDataset(Path(a.dataset) / "train", mode=0, scene_offset=offset).mean_cam_center.tolist()))
    return 0


def cmd_midpoint(a):
    import torch
    center = (torch.tensor(parse_xyz(a.a)) + torch.tensor(parse_xyz(a.b))) / 2   # float32, as the head stores it
    print(format_xyz(center.tolist()))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prereqs")
    for name in ("python", "datasets_dir", "scenes", "gpus", "encoder"):
        q.add_argument(f"--{name}", required=True)
    q.add_argument("--pose_version", choices=sorted(POSE_DIR_PREFIX), required=True)
    q.add_argument("--max_gpu_mem_used_mib", type=int, required=True)
    q.set_defaults(func=cmd_prereqs)
    q = sub.add_parser("env")
    q.add_argument("--gpu", required=True)
    q.add_argument("--provenance", action="store_true")
    q.set_defaults(func=cmd_env)
    q = sub.add_parser("write-settings")
    q.add_argument("path")
    q.add_argument("items", nargs="*")
    q.set_defaults(func=cmd_write_settings)
    q = sub.add_parser("head-record")
    for name in ("settings", "seed", "gpu", "head", "dataset", "command"):
        q.add_argument(f"--{name}", required=True)
    q.set_defaults(func=cmd_head_record)
    q = sub.add_parser("head-check")
    for name in ("settings", "seed", "gpu", "head", "out"):
        q.add_argument(f"--{name}", required=True)
    q.add_argument("--expected_sha256", default="")
    q.add_argument("--allow_mismatch", action="store_true")
    q.set_defaults(func=cmd_head_check)
    q = sub.add_parser("setup")
    q.add_argument("out")
    q.set_defaults(func=cmd_setup)
    q = sub.add_parser("run-summary")
    q.add_argument("run_dir")
    q.add_argument("--kind", choices=["stage_a", "replay", "none"], required=True)
    q.add_argument("--scenes", required=True)
    q.add_argument("--stage_a_dir", required=True)
    q.set_defaults(func=cmd_run_summary)
    q = sub.add_parser("summarize")
    q.add_argument("out")
    q.set_defaults(func=cmd_summarize)
    q = sub.add_parser("scene-center")
    q.add_argument("--dataset", required=True)
    q.add_argument("--offset", default="")
    q.set_defaults(func=cmd_scene_center)
    q = sub.add_parser("midpoint")
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.set_defaults(func=cmd_midpoint)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
