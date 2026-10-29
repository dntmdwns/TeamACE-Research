#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
from distutils.util import strtobool
from pathlib import Path

from ace_trainer import TrainerACE


def _strtobool(x):
    return bool(strtobool(x))


def _xyz(text):
    """'x,y,z' in meters -> [x, y, z] floats."""
    values = [float(v) for v in text.split(',')]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z, got {text!r}")
    return values


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('output_map_file', type=Path,
                        help='target file for the trained network')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--extra_scenes', type=Path, nargs='+', default=[],
                        help='additional scenes trained jointly with "scene" in one head; the training buffer is '
                             'split equally between all scenes and the head keeps the mean of "scene"')

    parser.add_argument('--resume_head', type=Path, default=None,
                        help='initialize the regression head from a previously trained ACE head; '
                             'the stored coordinate mean is preserved')

    parser.add_argument('--seed', type=int, default=2089,
                        help='base random seed used for buffer construction and training')

    parser.add_argument('--training_metrics_json', type=Path, default=None,
                        help='optional JSON file for timing, memory, and training configuration')

    parser.add_argument('--checkpoint_dir', type=Path, default=None,
                        help='optional directory for intermediate head checkpoints')

    parser.add_argument('--checkpoint_every', type=int, default=1,
                        help='save an intermediate head every N epochs when checkpoint_dir is set')

    parser.add_argument('--export_replay_buffer', type=Path, default=None,
                        help='optional file for exporting a random fixed-size feature replay buffer')

    parser.add_argument('--export_only', action='store_true',
                        help='build a feature buffer from a resumed trained head, export replay, and exit')

    parser.add_argument('--replay_buffer_features', type=int, default=200000,
                        help='number of features to export when export_replay_buffer is set')

    parser.add_argument('--replay_selection', choices=['random', 'filtered_random', 'voxel', 'quality', 'voxel_quality', 'voxel_feature_dedup', 'fps', 'voxel_weighted'], default='random',
                        help='method used to select exported replay features')

    parser.add_argument('--replay_feature_encoding', choices=['fp32', 'fp16', 'int8'], default='fp16',
                        help='storage encoding for exported replay features')

    parser.add_argument('--replay_dedup_cosine', type=float, default=0.98,
                        help='cosine threshold for same-voxel feature deduplication (0, 1]')
    parser.add_argument('--replay_dedup_pool_factor', type=int, default=4,
                        help='random shortlist size = min(valid candidates, factor * K)')

    parser.add_argument('--replay_voxel_size_m', type=float, default=0.25,
                        help='3D voxel size for coverage-aware replay selection')

    parser.add_argument('--replay_voxel_alpha', type=float, default=1.0,
                        help='voxel_weighted: draw voxels with probability proportional to (remaining candidates)^alpha; '
                             '0 weights every occupied voxel equally, 1 matches filtered_random')

    parser.add_argument('--replay_reprojection_error_px', type=float, default=5.0,
                        help='maximum L1 reprojection error for voxel replay candidates')

    parser.add_argument('--replay_drop_repeated_candidates', action='store_true',
                        help='drop repeated copies of the same (image, pixel) from the filtered candidate pool before '
                             'selection; the buffer samples pixels with replacement, so copies are identical entries')

    parser.add_argument('--replay_buffer', type=Path, default=None,
                        help='feature replay buffer exported by a previous scene')

    parser.add_argument('--replay_batch_mode', choices=['shuffle', 'stratified'], default='shuffle',
                        help='shuffle whole buffer or enforce an exact replay quota in each batch')

    parser.add_argument('--replay_ratio_percent', type=float, default=0.0,
                        help='percentage of current-scene training-buffer entries replaced by replay features')

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='lowest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='highest learning rate of 1 cycle scheduler')

    parser.add_argument('--training_buffer_size', type=int, default=8000000,
                        help='number of patches in the training buffer')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='number of patches drawn from each image when creating the buffer')

    parser.add_argument('--batch_size', type=int, default=5120,
                        help='number of patches for each parameter update (has to be a multiple of 512)')

    parser.add_argument('--epochs', type=int, default=16,
                        help='number of runs through the training buffer')

    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='minimum value of the soft clamping threshold when using a schedule')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')

    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=1000,
                        help='enforce maximum depth of network predictions')

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='split the training sequence in this number of clusters. disabled by default')

    parser.add_argument('--cluster_idx', type=int, default=None,
                        help='train on images part of this cluster. required only if --num_clusters is set.')

    # Params for the visualization. If enabled, it will slow down training considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_map_error_threshold', type=int, default=10,
                        help='reprojection error threshold for the visualisation in px')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--scene_offset', type=_xyz, default=None, metavar='X,Y,Z',
                        help='shift every camera position of the scene by this offset in meters (rotation unchanged), '
                             'to place scenes apart in one shared frame; off by default')

    parser.add_argument('--head_center', type=_xyz, default=None, metavar='X,Y,Z',
                        help='center (mean) of a fresh head; default: mean camera position of the (shifted) scene')

    parser.add_argument('--fp32_coordinates', action='store_true',
                        help='compute the head\'s final layer, dehomogenization and center addition in fp32 '
                             '(no autocast, no TF32) and save heads in fp32; for scenes far from the head center')

    parser.add_argument('--precision', choices=['legacy', 'fp32'], default='legacy',
                        help='fp32 disables AMP/TF32 and preserves FP32 replay and checkpoints')
    options = parser.parse_args()
    if options.precision == 'fp32':
        options.use_half = False
        options.replay_feature_encoding = 'fp32'

    trainer = TrainerACE(options)
    trainer.train()
