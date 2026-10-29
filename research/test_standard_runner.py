#!/usr/bin/env python3
"""End-to-end test of research/run_standard.sh without GPU jobs: stub train_ace.py/test_ace.py/dataset.py, a fake
nvidia-smi and a tmux that runs its windows synchronously. Covers stage-A training with head records, queueing several
seeds on one GPU, reusing heads (match, mismatch, missing record, override), overwrite refusal, prerequisite checks,
the shared-frame options (scene offsets, head center, fp32 coordinates) and voxel_weighted alphas."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

STUB_TRAIN = r'''
import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("scene"); p.add_argument("output_map_file")
for name in ("training_buffer_size", "batch_size", "epochs", "seed", "replay_buffer_features"):
    p.add_argument("--" + name, type=int)
for name in ("precision", "encoder_path", "resume_head", "export_replay_buffer", "replay_selection",
             "replay_reprojection_error_px", "replay_voxel_size_m", "replay_feature_encoding", "replay_buffer",
             "replay_ratio_percent", "replay_batch_mode", "training_metrics_json", "scene_offset", "head_center",
             "replay_voxel_alpha"):
    p.add_argument("--" + name)
p.add_argument("--export_only", action="store_true"); p.add_argument("--replay_drop_repeated_candidates", action="store_true")
p.add_argument("--fp32_coordinates", action="store_true")
a = p.parse_args()
if a.export_only:
    import torch
    k, g = a.replay_buffer_features, torch.Generator().manual_seed(a.seed)
    data = {"features": torch.randn(k, 512, generator=g).half(), "target_px": torch.randn(k, 2, generator=g),
            "gt_poses_inv": torch.randn(k, 4, 4, generator=g), "intrinsics": torch.randn(k, 3, 3, generator=g),
            "intrinsics_inv": torch.randn(k, 3, 3, generator=g)}
    meta = {"reprojection_error_px_l1": float(a.replay_reprojection_error_px), "valid_candidates": 1000,
            "repeated_candidates_dropped": 10 if a.replay_drop_repeated_candidates else None,
            "candidate_geometry_sha256": f"pool{a.seed}", "selected_occupied_voxels": 7, "selected_error_mean_px": 2.5}
    if a.replay_selection == "voxel_weighted":
        meta["voxel_alpha"] = float(a.replay_voxel_alpha or 1)
    torch.save({"selection": a.replay_selection, "selection_metadata": meta, "feature_encoding": a.replay_feature_encoding,
                "num_features": k, "data": data}, a.export_replay_buffer)
    Path(a.training_metrics_json).write_text(json.dumps({"replay_features_stored": k, "selection_metadata": meta}))
else:
    Path(a.output_map_file).write_bytes(f"head {a.scene} {a.seed} {a.resume_head}".encode())
    Path(a.training_metrics_json).write_text(json.dumps({
        "total_seconds": 10.0, "head_training_seconds": 8.0, "peak_gpu_memory_allocated_bytes": 2**30,
        "model_bytes": 100, "replay_ratio_percent": float(a.replay_ratio_percent or 0),
        "replay_batch_mode": a.replay_batch_mode or "shuffle", "replay_per_batch": 512 if a.replay_buffer else None,
        "replay_features_injected": 400000 if a.replay_buffer else 0}))
print("stub train", " ".join(sys.argv[1:]))
'''

STUB_TEST = r'''
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("scene"); p.add_argument("network")
for name in ("precision", "encoder_path", "session", "metrics_json", "scene_offset"):
    p.add_argument("--" + name)
p.add_argument("--fp32_coordinates", action="store_true"); p.add_argument("--region_centers", nargs="+")
a = p.parse_args()
acc = 90.0 if "chess" in a.scene and "stage_a" in a.session else 50.0
regions = {r.split("=")[0]: {"center": r.split("=")[1], "percent": 75.0 if r.split("=")[0] in a.scene else 25.0,
                             "median_distance_m": 0.5} for r in a.region_centers}
Path(a.metrics_json).write_text(json.dumps({"accuracy_percent": {"10cm_5deg": acc, "5cm_5deg": acc, "2cm_2deg": acc,
    "1cm_1deg": acc}, "median_translation_error_cm": 1.0, "median_rotation_error_deg": 0.5, "num_frames": 2,
    "nearest_region": regions}))
'''

# Mean camera position: a fixed per-scene point plus the offset (the real class averages the training poses).
STUB_DATASET = r'''
import torch
class CamLocDataset:
    def __init__(self, root_dir, mode=0, scene_offset=None):
        base = torch.tensor([0.5, -0.5, 1.0]) if "chess" in str(root_dir) else torch.tensor([1.5, -0.5, 1.0])
        self.mean_cam_center = base + (torch.zeros(3) if scene_offset is None else torch.tensor(scene_offset))
'''

FAKE_NVIDIA_SMI = r'''#!/usr/bin/env bash
case "$*" in
    *index,name,uuid,memory.used*) echo "0, Fake GPU, GPU-0, 1"; echo "1, Fake GPU, GPU-1, 1" ;;
    *compute-apps*) ;;
    *name,driver_version*) echo "Fake GPU, 1.0" ;;
    *name*) echo "Fake GPU" ;;
esac
'''

# Runs each window's command right away (without the trailing "; exec bash"), so launch returns when all are done.
FAKE_TMUX = r'''#!/usr/bin/env bash
case $1 in
    has-session) exit 1 ;;
    new-session|new-window) cmd=${@: -1}; bash -c "${cmd%; exec bash}" > /dev/null ;;
esac
'''


def write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


class StandardRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        write(self.repo / "train_ace.py", STUB_TRAIN)
        write(self.repo / "test_ace.py", STUB_TEST)
        write(self.repo / "ace_encoder_pretrained.pt", "encoder")
        for name in ("ace_trainer.py", "ace_network.py", "ace_loss.py", "ace_util.py", "precision.py",
                     "replay_buffer.py"):
            write(self.repo / name, f"# {name} fixture\n")
        write(self.repo / "dataset.py", STUB_DATASET)
        for name in ("run_standard.sh", "standard_runner.py"):
            (self.repo / "research").mkdir(exist_ok=True)
            shutil.copy(HERE / name, self.repo / "research" / name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.commit("fixture")
        bin_dir = self.tmp / "bin"
        write(bin_dir / "nvidia-smi", FAKE_NVIDIA_SMI, True)
        write(bin_dir / "tmux", FAKE_TMUX, True)
        stubs = self.tmp / "stubs"
        write(stubs / "cv2.py", "__version__ = 'stub'\n")
        write(stubs / "dsacstar.py", "")
        # CUDA is reported available only to the prerequisite probe (the one call that imports dsacstar).
        write(stubs / "cuda/sitecustomize.py", "import torch\ntorch.cuda.is_available = lambda: True\n")
        write(bin_dir / "python", f'#!/usr/bin/env bash\npath={stubs}; [[ $1 == -c && $2 == *dsacstar* ]] && '
              f'path={stubs}:{stubs}/cuda\nPYTHONPATH=$path exec {sys.executable} "$@"\n', True)
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
        self.datasets = self.tmp / "datasets"
        for scene, focal in (("chess", "526.2"), ("fire", "524.9")):
            for split in ("train", "test"):
                for sub, text in (("rgb", "x"), ("poses", "1 0 0 0"), ("calibration", focal)):
                    for frame in range(2):
                        write(self.datasets / f"pgt_7scenes_{scene}" / split / sub / f"f{frame}.txt", text)

    def commit(self, message):
        git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-qm", message], check=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def runner(self, *args):
        command = ["bash", str(self.repo / "research/run_standard.sh"), *args]
        return subprocess.run(command, capture_output=True, text=True, env=self.env, cwd=self.tmp)

    def launch(self, out, *args):
        return self.runner("launch", "--out", str(self.tmp / out), "--python", str(self.tmp / "bin/python"),
                           "--datasets_dir", str(self.datasets), *args)

    def test_train_queue_reuse_and_refusals(self):
        # Two seeds queued on one GPU, stage A trained per seed, no replay plus two K values.
        r = self.launch("a", "--seeds", "2089,2090", "--gpus", "0", "--methods", "none,filtered_random", "--ks", "5000,2000")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = self.tmp / "a"
        console = (out / "logs/gpu0.console.log").read_text()
        self.assertLess(console.index("stage_a seed2089 OK"), console.index("stage_a seed2090 OK"))
        self.assertNotIn("FAILED", console)
        for seed in (2089, 2090):
            record = json.loads((out / f"stage_a/seed{seed}/stage_a_head.json").read_text())
            self.assertEqual(record["settings"]["seed"], seed)
            self.assertEqual(record["settings"]["pose_version"], "pgt")
            self.assertEqual(record["settings"]["gpu_name"], "Fake GPU")
            for run in ("none", "filtered_random/K5000", "filtered_random/K2000"):
                provenance = (out / run / f"seed{seed}/provenance.txt").read_text()
                self.assertIn("exit_status: 0", provenance)
                self.assertIn(f"chess_head_sha256: {record['head_sha256']}", provenance)
                self.assertTrue((out / run / f"seed{seed}/eval_stage_b_chess.json").is_file())
            commands = (out / f"filtered_random/K5000/seed{seed}/commands.txt").read_text()
            self.assertIn("OMP_NUM_THREADS=1", commands)
            self.assertIn("--replay_drop_repeated_candidates", commands)
            self.assertIn("--replay_batch_mode stratified", commands)
        summary = (out / "summary.md").read_text()
        self.assertIn("| filtered_random, K = 2,000 | chess | 2 | 50.0 ± 0.0", summary)
        self.assertIn("| 40.0 ± 0.0 |", summary)   # forgetting: 90 before, 50 after
        self.assertIn("`seeds` = 2089,2090 (standard: 2089,2090,2091)", (out / "SETUP.md").read_text())

        # Refuse to overwrite.
        r = self.launch("a", "--seeds", "2089,2090", "--gpus", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing to overwrite", r.stderr)

        heads = str(out / "stage_a/seed{seed}/stage_a_head.pt")
        record = json.loads((out / "stage_a/seed2089/stage_a_head.json").read_text())
        self.assertEqual(sorted(record["settings"]["training_code_sha256"]),
                         ["ace_loss.py", "ace_network.py", "ace_trainer.py", "ace_util.py", "dataset.py",
                          "precision.py", "replay_buffer.py", "train_ace.py"])
        self.assertNotIn("git_commit", record["settings"])
        self.assertEqual(len(record["git_commit"]), 40)   # recorded, not compared
        # A doc-only commit keeps the heads valid: reuse without override; no stage-A training in the new folder.
        write(self.repo / "README.md", "docs\n")
        self.commit("docs only")
        r = self.launch("b", "--seeds", "2089,2090", "--gpus", "0,1", "--stage_a_head", heads, "--export_only")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse((self.tmp / "b/stage_a/seed2089/stage_a_head.pt").exists())
        self.assertEqual(json.loads((self.tmp / "b/stage_a/seed2090/head_check.json").read_text())["status"], "match")
        self.assertIn("seed2090", (self.tmp / "b/logs/gpu1.console.log").read_text())

        # A different setting is refused, listing the mismatch; the override records it in SETUP.md.
        r = self.launch("c", "--seeds", "2089,2090", "--gpus", "0", "--stage_a_head", heads, "--epochs", "8")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("epochs: head 16, this run 8", r.stdout)
        self.assertFalse((self.tmp / "c").exists())
        r = self.launch("c", "--seeds", "2089,2090", "--gpus", "0", "--stage_a_head", heads, "--epochs", "8",
                        "--allow_head_mismatch", "--head_mismatch_reason", "only fewer epochs, for the test",
                        "--export_only")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        setup = (self.tmp / "c/SETUP.md").read_text()
        self.assertIn("`epochs` = 8 (standard: 16)", setup)
        self.assertIn("used with `--allow_head_mismatch`: epochs: head 16, this run 8", setup)
        self.assertIn("this run 8. Reason: only fewer epochs, for the test", setup)
        r = self.launch("c2", "--seeds", "2089", "--gpus", "0", "--head_mismatch_reason", "x")
        self.assertIn("needs --allow_head_mismatch", r.stderr)

        # A commit that changes a training file is refused, naming the file.
        write(self.repo / "ace_trainer.py", "# changed training code\n")
        self.commit("training change")
        r = self.launch("e", "--seeds", "2089,2090", "--gpus", "0", "--stage_a_head", heads, "--export_only")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("training_code_sha256.ace_trainer.py: head", r.stdout)
        self.assertNotIn("training_code_sha256.dataset.py", r.stdout)
        self.assertFalse((self.tmp / "e").exists())
        # So is a change to a file the trainer imports (the loss), with only that file named beyond the first.
        write(self.repo / "ace_loss.py", "# changed loss\n")
        self.commit("loss change")
        r = self.launch("e", "--seeds", "2089,2090", "--gpus", "0", "--stage_a_head", heads, "--export_only")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("training_code_sha256.ace_loss.py: head", r.stdout)
        self.assertNotIn("training_code_sha256.replay_buffer.py", r.stdout)

        # A missing record is refused; with the override a sha256 is still required.
        (out / "stage_a/seed2089/stage_a_head.json").unlink()
        r = self.launch("d", "--seeds", "2089", "--gpus", "0", "--stage_a_head", heads, "--allow_head_mismatch")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no sha256 to check the head against", r.stdout)
        r = self.launch("d", "--seeds", "2089", "--gpus", "0", "--stage_a_head", heads, "--allow_head_mismatch",
                        "--stage_a_head_sha256", "2089=" + "0" * 64)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("!= expected", r.stdout)

    def test_voxel_weighted_alphas(self):
        # One run per alpha in its own folder; other methods get no alpha argument.
        r = self.launch("w", "--seeds", "2089", "--gpus", "0", "--methods", "filtered_random,voxel_weighted",
                        "--voxel_alphas", "0,0.25,1", "--ks", "5000")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("(voxel_weighted x alpha 0 0.25 1)", r.stdout)
        out = self.tmp / "w"
        self.assertNotIn("FAILED", (out / "logs/gpu0.console.log").read_text())
        self.assertNotIn("--replay_voxel_alpha", (out / "filtered_random/K5000/seed2089/commands.txt").read_text())
        for alpha in ("0", "0.25", "1"):
            run = out / f"voxel_weighted_alpha{alpha}/K5000/seed2089"
            self.assertIn(f"--replay_selection voxel_weighted", (run / "commands.txt").read_text())
            self.assertIn(f"--replay_voxel_alpha {alpha} ", (run / "commands.txt").read_text())
            self.assertEqual(json.loads((run / "summary.json").read_text())["replay_store"]["voxel_alpha"], float(alpha))
        summary = (out / "summary.md").read_text()
        labels = [f"| voxel_weighted (alpha = {a}), K = 5,000 |" for a in ("0", "0.25", "1")]
        self.assertTrue(all(label in summary for label in labels))
        self.assertLess(summary.index(labels[1]), summary.index(labels[2]))
        self.assertIn("voxel_weighted (alpha = 0.25)", (out / "summary.csv").read_text())
        self.assertIn("filtered_random,voxel_weighted; K = 5000; voxel_weighted alpha = 0,0.25,1",
                      (out / "SETUP.md").read_text())
        self.assertEqual(json.loads((out / "settings.json").read_text())["voxel_alphas"], "0,0.25,1")
        # Alphas without the method, negative or duplicate alphas are refused.
        for args, message in ((["--voxel_alphas", "0.5"], "applies to method voxel_weighted only"),
                              (["--methods", "voxel_weighted", "--voxel_alphas", "-1"], "non-negative numbers"),
                              (["--methods", "voxel_weighted", "--voxel_alphas", "0,0"], "has duplicates")):
            r = self.launch("x", "--seeds", "2089", "--gpus", "0", *args)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn(message, r.stderr)
            self.assertFalse((self.tmp / "x").exists())

    def test_coordinate_frame_options(self):
        new_flags = ("--scene_offset", "--head_center", "--fp32_coordinates")
        # Off by default: no training or evaluation command gets a new option; evaluation reports regions.
        # (commands.txt quotes with printf %q, which escapes commas; backslashes are dropped before matching.)
        r = self.launch("std", "--seeds", "2089", "--gpus", "0", "--methods", "none,filtered_random")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = self.tmp / "std"
        for run in ("stage_a", "none", "filtered_random/K5000"):
            commands = (out / run / "seed2089/commands.txt").read_text().replace("\\", "")
            for flag in new_flags:
                self.assertNotIn(flag, commands, run)
            self.assertIn("--region_centers chess=0.5,-0.5,1.0 fire=1.5,-0.5,1.0", commands)
        setup = (out / "SETUP.md").read_text()
        self.assertIn("offsets chess none, fire none", setup)
        self.assertIn("Deviations from the standard setup:\n\n- `seeds`", setup)
        self.assertIn("| no replay | chess | 1 | 50.0 | 50.0 | 50.0 | 50.0 | 40.0 | chess 75.0, fire 25.0 |",
                      (out / "summary.md").read_text())
        record = json.loads((out / "stage_a/seed2089/stage_a_head.json").read_text())["settings"]
        self.assertEqual((record["scene_offset"], record["head_center"], record["fp32_coordinates"]), ("", "scene_a", "0"))

        # Separated scenes: B shifted, head centered between the shifted scene centers, fp32 coordinates.
        r = self.launch("sep", "--seeds", "2089", "--gpus", "0", "--offset_a", "-4,0,0", "--offset_b", "4,0,0",
                        "--head_center", "midpoint", "--fp32_coordinates", "1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = self.tmp / "sep"
        stage_a = (out / "stage_a/seed2089/commands.txt").read_text().replace("\\", "").splitlines()
        self.assertIn("--scene_offset=-4,0,0 --head_center=1.0,-0.5,1.0 --seed 2089", stage_a[0])
        self.assertIn("--fp32_coordinates", stage_a[0])
        self.assertIn("--session stage_a_chess --metrics_json", stage_a[2])
        self.assertIn("--scene_offset=-4,0,0 --fp32_coordinates --region_centers chess=-3.5,-0.5,1.0 fire=5.5,-0.5,1.0",
                      stage_a[2])
        self.assertIn("--scene_offset=4,0,0 --fp32_coordinates", stage_a[3])
        export, train_b, eval_a, eval_b = (out / "filtered_random/K5000/seed2089/commands.txt").read_text().replace(
            "\\", "").splitlines()
        self.assertIn("--fp32_coordinates --scene_offset=-4,0,0 --seed 2089 --resume_head", export)
        self.assertIn("--fp32_coordinates --scene_offset=4,0,0 --seed 2089 --resume_head", train_b)
        self.assertNotIn("--head_center", export + train_b)   # stage B keeps the stage-A head's center
        self.assertIn("--scene_offset=-4,0,0", eval_a)
        self.assertIn("--scene_offset=4,0,0", eval_b)
        setup = (out / "SETUP.md").read_text()
        for line in ("`offset_a` = -4,0,0 (standard: off)", "`offset_b` = 4,0,0 (standard: off)",
                     "`head_center` = midpoint (standard: scene_a)", "`fp32_coordinates` = 1 (standard: 0)",
                     "midpoint of both scenes' mean camera positions (1.0,-0.5,1.0)"):
            self.assertIn(line, setup)
        record = json.loads((out / "stage_a/seed2089/stage_a_head.json").read_text())["settings"]
        self.assertEqual((record["scene_offset"], record["head_center"], record["fp32_coordinates"]),
                         ("-4,0,0", "1.0,-0.5,1.0", "1"))

        # Reuse: a record written before these settings existed counts as the defaults; any other frame is refused.
        heads = str(self.tmp / "std/stage_a/seed{seed}/stage_a_head.pt")
        record_path = self.tmp / "std/stage_a/seed2089/stage_a_head.json"
        record = json.loads(record_path.read_text())
        for key in ("scene_offset", "head_center", "fp32_coordinates"):
            del record["settings"][key]
        record_path.write_text(json.dumps(record))
        r = self.launch("reuse", "--seeds", "2089", "--gpus", "0", "--stage_a_head", heads, "--export_only")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.launch("reuse2", "--seeds", "2089", "--gpus", "0", "--stage_a_head", heads, "--offset_a", "8,0,0",
                        "--fp32_coordinates", "1", "--export_only")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("scene_offset: head '', this run '8,0,0'", r.stdout)
        self.assertIn("fp32_coordinates: head '0', this run '1'", r.stdout)
        r = self.launch("reuse3", "--seeds", "2089", "--gpus", "0", "--stage_a_head", heads, "--head_center", "midpoint")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--head_center applies to a fresh stage-A head", r.stderr)
        r = self.launch("bad", "--seeds", "2089", "--gpus", "0", "--offset_b", "8,0")
        self.assertIn("--offset_b takes X,Y,Z", r.stderr)

    def test_prerequisites(self):
        shutil.rmtree(self.datasets / "pgt_7scenes_fire")
        for f in (self.datasets / "pgt_7scenes_chess/test/calibration").iterdir():
            f.write_text("525.0")
        r = self.runner("check", "--out", str(self.tmp / "x"), "--python", str(self.tmp / "bin/python"),
                        "--datasets_dir", str(self.datasets), "--gpus", "0,5")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("pgt_7scenes_fire: missing", r.stdout)
        self.assertIn("poses look calibrated", r.stdout)
        self.assertIn("GPU 5 not found", r.stdout)
        self.assertFalse((self.tmp / "x").exists())


if __name__ == "__main__":
    unittest.main()
