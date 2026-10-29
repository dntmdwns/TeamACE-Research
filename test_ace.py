#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import json
import logging
import math
import random
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset

import dsacstar
from ace_network import Regressor
from precision import configure_precision, precision_metadata
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def _xyz(text):
    """'x,y,z' in meters -> [x, y, z] floats."""
    values = [float(v) for v in text.split(',')]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z, got {text!r}")
    return values


def _region(text):
    """'name=x,y,z' -> (name, [x, y, z])."""
    name, sep, xyz = text.partition('=')
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected name=x,y,z, got {text!r}")
    return name, _xyz(xyz)


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--metrics_json', type=Path, default=None,
                        help='optional JSON file containing all aggregate localization metrics')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='use FP16 autocast during evaluation')
    parser.add_argument('--seed', type=int, default=2089,
                        help='seed Python, NumPy, and Torch for evaluation')

    parser.add_argument('--max_test_images', type=int, default=None,
                        help='evaluate an evenly spaced deterministic subset of at most this many test images')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # ACE is RGB-only, no need for this param.
    # parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2], help='test mode: 1 = RGB, 2 = RGB-D')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False,
                        help='set to true if your queries are not a smooth video')

    parser.add_argument('--render_pose_error_threshold', type=int, default=20,
                        help='pose error threshold for the visualisation in cm/deg')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_frame_skip', type=int, default=1,
                        help='skip every xth frame for long and dense query sequences')

    parser.add_argument('--scene_offset', type=_xyz, default=None, metavar='X,Y,Z',
                        help='shift the ground truth camera positions by this offset (as in training); off by default')

    parser.add_argument('--fp32_coordinates', action='store_true',
                        help='compute the head\'s final layer and center addition in fp32 (as in training)')

    parser.add_argument('--region_centers', type=_region, nargs='+', default=None, metavar='NAME=X,Y,Z',
                        help='diagnostic: count test frames whose estimated camera position is nearest to each of '
                             'these centers (e.g. the shifted mean training camera position of every scene)')

    parser.add_argument('--precision', choices=['legacy', 'fp32'], default='legacy')
    opt = parser.parse_args()
    if opt.precision == 'fp32':
        opt.use_half = False
    configure_precision(opt.precision)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        use_half=opt.use_half,
        image_height=opt.image_resolution,
        scene_offset=opt.scene_offset,
    )
    if opt.max_test_images is not None and opt.max_test_images < len(testset):
        if opt.max_test_images < 1:
            parser.error('--max_test_images must be at least 1')
        subset_indices = np.linspace(0, len(testset) - 1, opt.max_test_images, dtype=int).tolist()
        testset = Subset(testset, subset_indices)
        _logger.info('Using an evenly spaced diagnostic subset of %d test images.', len(testset))
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network.heads.fp32_coordinates = opt.fp32_coordinates

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    if opt.region_centers is not None:
        region_names = [name for name, _ in opt.region_centers]
        region_centers_R3 = torch.tensor([xyz for _, xyz in opt.region_centers], dtype=torch.float64)
        region_counts = {name: 0 for name in region_names}
        region_distances = {name: [] for name in region_names}

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = vutil.get_rendering_target_path(
            opt.render_target_path,
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=opt.use_half,
            image_height=opt.image_resolution,
        )

        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # Testing loop.
    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=opt.use_half):
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(
                    zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):

                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))

                # Compute the pose via RANSAC.
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                if opt.region_centers is not None:
                    # Diagnostic only: the region whose center is nearest to the estimated camera position.
                    distances = torch.norm(region_centers_R3 - out_pose[0:3, 3].double(), dim=1)
                    nearest = region_names[int(distances.argmin())]
                    region_counts[nearest] += 1
                    region_distances[nearest].append(float(distances.min()))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=opt.render_sparse_queries)

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                # Check various thresholds.
                if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                    pct1 += 1

                # Write estimated pose to pose file (inverse).
                out_pose = out_pose.inverse()

                # Translation.
                t = out_pose[0:3, 3]

                # Rotation to axis angle.
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle

                # Axis angle to quaternion.
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                # Write to output file. All in a single line.
                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info('Accuracy:')
    _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    if opt.metrics_json is not None:
        opt.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "scene": str(scene_path),
            "network": str(head_network_path),
            "session": opt.session,
            "num_frames": total_frames,
            "precision": precision_metadata(opt.precision, opt.use_half),
            "accuracy_percent": {
                "10cm_5deg": pct10_5,
                "5cm_5deg": pct5,
                "2cm_2deg": pct2,
                "1cm_1deg": pct1,
            },
            "median_rotation_error_deg": float(median_rErr),
            "median_translation_error_cm": float(median_tErr),
            "average_processing_time_ms": avg_time * 1000,
        }
        if opt.scene_offset is not None or opt.fp32_coordinates:
            metrics["scene_offset"] = opt.scene_offset
            metrics["fp32_coordinates"] = opt.fp32_coordinates
            metrics["head_center"] = network.heads.mean.flatten().tolist()
        if opt.region_centers is not None:
            metrics["nearest_region"] = {
                name: {"center": xyz, "percent": region_counts[name] / total_frames * 100,
                       "median_distance_m": float(np.median(region_distances[name])) if region_distances[name] else None}
                for name, xyz in opt.region_centers}
        with opt.metrics_json.open("w") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
            metrics_file.write("\n")
        _logger.info("Saved aggregate metrics to: %s", opt.metrics_json)

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")

    test_log.close()
    pose_log.close()
