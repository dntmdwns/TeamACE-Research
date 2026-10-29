# Copyright © Niantic, Inc. 2022.

import json
import hashlib
import logging
import math
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from fps_sampling import farthest_point_sample
from precision import configure_precision, precision_metadata
from dataset import CamLocDataset
from replay_buffer import (
    FP16_ENCODING,
    FP32_ENCODING,
    INT8_ENCODING,
    balanced_voxel_sample,
    voxel_feature_dedup_sample,
    weighted_voxel_sample,
    stratified_epoch_indices,
    dequantize_features_int8,
    quantize_features_int8,
)

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE:
    def __init__(self, options):
        self.options = options
        configure_precision(options.precision)

        self.device = torch.device('cuda')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = self.options.seed
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        # Kept separate so exporting/sampling replay does not change the ordinary training order.
        self.replay_generator = torch.Generator()
        self.replay_generator.manual_seed(self.base_seed + 12289)

        if not 0 <= self.options.replay_ratio_percent < 100:
            raise ValueError("replay_ratio_percent must be in [0, 100)")
        if self.options.replay_ratio_percent > 0 and self.options.replay_buffer is None:
            raise ValueError("replay_buffer is required when replay_ratio_percent is greater than zero")
        if self.options.replay_buffer_features < 1:
            raise ValueError("replay_buffer_features must be at least 1")
        if self.options.replay_voxel_size_m <= 0:
            raise ValueError("replay_voxel_size_m must be positive")
        if self.options.replay_reprojection_error_px <= 0:
            raise ValueError("replay_reprojection_error_px must be positive")
        if (not 0 < self.options.replay_dedup_cosine <= 1 or
                self.options.replay_dedup_pool_factor < 1):
            raise ValueError('Invalid dedup cosine threshold or shortlist pool factor')
        self.voxel_alpha = getattr(self.options, 'replay_voxel_alpha', 1.0)
        if not math.isfinite(self.voxel_alpha) or self.voxel_alpha < 0:
            raise ValueError("replay_voxel_alpha must be finite and non-negative")
        self.drop_repeated_candidates = getattr(self.options, 'replay_drop_repeated_candidates', False)
        if self.drop_repeated_candidates and self.options.replay_selection == "random":
            raise ValueError("replay_drop_repeated_candidates requires a filtered replay_selection")
        if self.options.export_only:
            if self.options.resume_head is None or self.options.export_replay_buffer is None:
                raise ValueError("export_only requires resume_head and export_replay_buffer")
            if self.options.replay_ratio_percent > 0:
                raise ValueError("export_only cannot inject a replay buffer")
        if (self.options.replay_selection != "random" and
                self.options.export_replay_buffer is not None and
                self.options.replay_ratio_percent > 0):
            raise ValueError(
                "Filtered export after replay injection is not supported yet; export the current scene in a separate run"
            )

        self.replay_buffer_bytes = 0
        self.replay_features_injected = 0
        self.replay_features_stored = 0
        self.replay_selection = None
        self.replay_feature_encoding = None

        self.iteration = 0
        self.optimizer_steps = 0  # Optimizer steps actually taken (GradScaler can skip some).
        self.training_start = None
        self.num_data_loader_workers = 12

        # Optional shared coordinate frame (all off by default): the scene's camera positions shifted by a fixed offset,
        # an explicit head center for a fresh head, and fp32 scene coordinates in the head's final layer.
        self.scene_offset = getattr(self.options, 'scene_offset', None)
        self.head_center = getattr(self.options, 'head_center', None)
        self.fp32_coordinates = getattr(self.options, 'fp32_coordinates', False)
        if self.head_center is not None and self.options.resume_head is not None:
            raise ValueError("head_center sets the center of a fresh head; a resumed head keeps its stored center")

        # Joint training: extra scenes fill equal shares of the buffer; the head keeps the first scene's mean.
        self.extra_scenes = list(getattr(self.options, 'extra_scenes', None) or [])
        if self.extra_scenes and self.scene_offset is not None:
            raise ValueError("scene_offset is not supported with extra_scenes")
        if self.extra_scenes:
            if self.options.training_buffer_size % (1 + len(self.extra_scenes)):
                raise ValueError("training_buffer_size must be divisible by the number of scenes")
            if (self.options.replay_ratio_percent > 0 or self.options.export_replay_buffer is not None or
                    self.options.render_visualization):
                raise ValueError("extra_scenes does not support replay, replay export, or visualization")

        def make_dataset(scene):
            return CamLocDataset(
                root_dir=scene / "train",
                mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=self.options.use_aug,
                aug_rotation=self.options.aug_rotation,
                aug_scale_max=self.options.aug_scale,
                aug_scale_min=1 / self.options.aug_scale,
                num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
                cluster_idx=self.options.cluster_idx,    # Optional clustering for Cambridge experiments.
                scene_offset=self.scene_offset,
            )

        # Create dataset(s).
        self.scenes = [self.options.scene] + self.extra_scenes
        self.datasets = [make_dataset(scene) for scene in self.scenes]
        self.dataset = self.datasets[0]

        for scene, dataset in zip(self.scenes, self.datasets):
            _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
                scene,
                len(dataset),
                dataset.mean_cam_center[0],
                dataset.mean_cam_center[1],
                dataset.mean_cam_center[2]))

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center if self.head_center is None else torch.tensor(self.head_center),
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous
        )
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        if self.options.resume_head is not None:
            head_state_dict = torch.load(self.options.resume_head, map_location="cpu")
            self.regressor.heads.load_state_dict(head_state_dict, strict=True)
            _logger.info(
                "Continued the same regression head from: %s (stored mean: %.3f %.3f %.3f)",
                self.options.resume_head,
                *self.regressor.heads.mean.flatten().tolist(),
            )

        if self.head_center is not None:
            _logger.info("Fresh head with explicit center: %.3f %.3f %.3f", *self.regressor.heads.mean.flatten().tolist())
        if self.fp32_coordinates:
            self.regressor.heads.fp32_coordinates = True
            _logger.info("fp32 scene coordinates in the head's final layer; heads are saved in fp32.")

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)

        # Setup learning rate scheduler.
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        if self.options.render_visualization:
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.options.render_target_path,
                self.options.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold)
        else:
            self.ace_visualizer = None

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.options.render_camera_z_offset
            )

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        if self.options.export_only:
            self.export_replay_buffer()
            buffer_end_time = time.time()
            creating_buffer_time += buffer_end_time - buffer_start_time
            total_time = time.time() - self.training_start
            _logger.info(
                "Replay export completed without head training in %.1f seconds.",
                total_time,
            )
            if self.options.training_metrics_json is not None:
                self.options.training_metrics_json.parent.mkdir(parents=True, exist_ok=True)
                metrics = {
                    "scene": str(self.options.scene),
                    "resume_head": str(self.options.resume_head),
                    "seed": self.base_seed,
                    "export_only": True,
                    "precision": precision_metadata(self.options.precision, self.options.use_half),
                    "training_buffer_size": self.options.training_buffer_size,
                    "buffer_creation_and_export_seconds": creating_buffer_time,
                    "total_seconds": total_time,
                    "replay_features_stored": self.replay_features_stored,
                    "replay_buffer_bytes": self.replay_buffer_bytes,
                    "replay_selection": self.replay_selection,
                    "replay_feature_encoding": self.replay_feature_encoding,
                    "selection_metadata": getattr(self, "selection_diagnostics", {}),
                    **self.coordinate_frame_metadata(),
                }
                with self.options.training_metrics_json.open("w") as metrics_file:
                    json.dump(metrics, metrics_file, indent=2)
                    metrics_file.write("\n")
            return

        # Random export does not depend on the trained head. Keeping it here
        # preserves the exact behavior of the established random baseline.
        if (self.options.export_replay_buffer is not None and
                self.options.replay_selection == "random"):
            self.export_replay_buffer()
        if self.options.replay_ratio_percent > 0:
            self.inject_replay_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        # Train the regression head.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            self.run_epoch()
            training_time += time.time() - epoch_start_time
            completed_epoch = self.epoch + 1
            if (self.options.checkpoint_dir is not None and
                    completed_epoch % self.options.checkpoint_every == 0):
                checkpoint_path = self.options.checkpoint_dir / f"epoch_{completed_epoch:03d}.pt"
                self.save_model(checkpoint_path)

        # Geometry-aware selection needs predictions from the trained head.
        if (self.options.export_replay_buffer is not None and
                self.options.replay_selection != "random"):
            self.export_replay_buffer()

        # Save trained model.
        self.save_model()

        end_time = time.time()
        total_time = end_time - self.training_start
        peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {total_time:.1f} seconds. '
                     f'Peak allocated GPU memory: {peak_allocated_bytes / 1024 ** 3:.2f}GB.')

        if self.options.training_metrics_json is not None:
            self.options.training_metrics_json.parent.mkdir(parents=True, exist_ok=True)
            metrics = {
                "scene": str(self.options.scene),
                "output_map_file": str(self.options.output_map_file),
                "resume_head": str(self.options.resume_head) if self.options.resume_head is not None else None,
                "seed": self.base_seed,
                "training_buffer_size": self.options.training_buffer_size,
                "samples_per_image": self.options.samples_per_image,
                "batch_size": self.options.batch_size,
                "epochs": self.options.epochs,
                "buffer_creation_seconds": creating_buffer_time,
                "head_training_seconds": training_time,
                "total_seconds": total_time,
                "peak_gpu_memory_allocated_bytes": peak_allocated_bytes,
                "peak_gpu_memory_reserved_bytes": peak_reserved_bytes,
                "precision": precision_metadata(self.options.precision, self.options.use_half),
                "model_bytes": self.options.output_map_file.stat().st_size,
                "replay_ratio_percent": self.options.replay_ratio_percent,
                "replay_batch_mode": self.options.replay_batch_mode,
                "replay_per_batch": (round(self.options.batch_size * self.options.replay_ratio_percent / 100)
                                     if self.options.replay_batch_mode == "stratified" else None),
                "replay_features_injected": self.replay_features_injected,
                "replay_features_stored": self.replay_features_stored,
                "replay_buffer_bytes": self.replay_buffer_bytes,
                "replay_selection": self.replay_selection,
                "replay_feature_encoding": self.replay_feature_encoding,
            }
            metrics.update(self.coordinate_frame_metadata())
            if self.extra_scenes:
                metrics["extra_scenes"] = [str(scene) for scene in self.extra_scenes]
                metrics["buffer_entries_per_scene"] = self.options.training_buffer_size // len(self.scenes)
            with self.options.training_metrics_json.open("w") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)
                metrics_file.write("\n")
            _logger.info("Saved training metrics to: %s", self.options.training_metrics_json)

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                mode=0,
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=False) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)

    def coordinate_frame_metadata(self):
        """Shared-frame settings for the metrics JSON; empty when all are off, so default runs record nothing new."""
        if self.scene_offset is None and self.head_center is None and not self.fp32_coordinates:
            return {}
        return {"scene_offset": self.scene_offset, "head_center": self.regressor.heads.mean.flatten().tolist(),
                "fp32_coordinates": self.fp32_coordinates}

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device)
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Each scene fills its own consecutive share of the buffer (the whole buffer for a single scene).
            scene_share = self.options.training_buffer_size // len(self.datasets)
            for scene_idx, dataset in enumerate(self.datasets):
                scene_end = (scene_idx + 1) * scene_share

                # Sampler.
                batch_sampler = sampler.BatchSampler(sampler.RandomSampler(dataset, generator=self.batch_generator),
                                                     batch_size=1,
                                                     drop_last=False)

                # Used to seed workers in a reproducible manner.
                def seed_worker(worker_id):
                    # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
                    # the dataloader generator.
                    worker_seed = torch.initial_seed() % 2 ** 32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)

                # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
                # need to rescale all images in the batch to the same size).
                training_dataloader = DataLoader(dataset=dataset,
                                                 sampler=batch_sampler,
                                                 batch_size=None,
                                                 worker_init_fn=seed_worker,
                                                 generator=self.loader_generator,
                                                 pin_memory=True,
                                                 num_workers=self.num_data_loader_workers,
                                                 persistent_workers=self.num_data_loader_workers > 0,
                                                 timeout=60 if self.num_data_loader_workers > 0 else 0,
                                                 )

                # Iterate until the training buffer is full.
                buffer_idx = scene_idx * scene_share
                dataset_passes = 0

                while buffer_idx < scene_end:
                    dataset_passes += 1
                    for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:

                        # Copy to device.
                        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                        image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                        gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                        intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                        intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                        # Compute image features.
                        with autocast(enabled=self.options.use_half):
                            features_BCHW = self.regressor.get_features(image_B1HW)

                        # Dimensions after the network's downsampling.
                        B, C, H, W = features_BCHW.shape

                        # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                        image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                        image_mask_B1HW = image_mask_B1HW.bool()

                        # If the current mask has no valid pixels, continue.
                        if image_mask_B1HW.sum() == 0:
                            continue

                        # Create a tensor with the pixel coordinates of every feature vector.
                        pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                        pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                        pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                        # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                        gt_pose_inv = gt_pose_inv_B44[:, :3]
                        gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                        # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                        intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                        intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                        def normalize_shape(tensor_in):
                            """Bring tensor from shape BxCxHxW to NxC"""
                            return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                        batch_data = {
                            'features': normalize_shape(features_BCHW),
                            'target_px': normalize_shape(pixel_positions_B2HW),
                            'gt_poses_inv': gt_pose_inv,
                            'intrinsics': intrinsics,
                            'intrinsics_inv': intrinsics_inv
                        }

                        # Turn image mask into sampling weights (all equal).
                        image_mask_B1HW = image_mask_B1HW.float()
                        image_mask_N1 = normalize_shape(image_mask_B1HW)

                        # Over-sample according to image mask.
                        features_to_select = self.options.samples_per_image * B
                        features_to_select = min(features_to_select, scene_end - buffer_idx)

                        # Sample indices uniformly, with replacement.
                        sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                        features_to_select,
                                                        replacement=True,
                                                        generator=self.sampling_generator)

                        # Select the data to put in the buffer.
                        for k in batch_data:
                            batch_data[k] = batch_data[k][sample_idxs]

                        # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                        buffer_offset = buffer_idx + features_to_select
                        for k in batch_data:
                            self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                        buffer_idx = buffer_offset
                        if buffer_idx >= scene_end:
                            break

                if len(self.datasets) > 1:
                    _logger.info(f"Filled buffer entries {scene_end - scene_share}-{scene_end - 1} from "
                                 f"{self.scenes[scene_idx]} with {dataset_passes} passes.")

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def _first_copy_positions(self, candidate_indices):
        """Ascending positions (CPU) of the first copy of every candidate, dropping repeated copies.

        The buffer samples pixels with replacement, so one (image, pixel) can fill several entries with identical
        features, target pixel, pose and intrinsics. Entries are hashed over the raw bits of all four fields and
        equal hashes are confirmed by exact comparison, so a hash collision never drops a distinct entry.
        Everything runs on the GPU and returns integer positions: CPU boolean-mask indexing of tensors this large
        crashed intermittently with PyTorch 2.4.1 here (nonzero INTERNAL ASSERT, heap corruption).
        """
        fields = ('features', 'target_px', 'gt_poses_inv', 'intrinsics')
        candidates = candidate_indices.to(self.device)

        def raw_bits(rows):
            parts = [self.training_buffer[key][rows].flatten(1) for key in fields]
            return torch.cat([part.view(torch.int16 if part.dtype == torch.float16 else torch.int32).long()
                              for part in parts], dim=1)

        # Fixed generator: hashing must not consume the replay generator, so selections stay reproducible.
        width = raw_bits(candidates[:1]).shape[1]
        coefficients = (torch.randint(0, 2 ** 62, (width,), generator=torch.Generator().manual_seed(0)) * 2 + 1)
        coefficients = coefficients.to(self.device)
        hashes = torch.cat([(raw_bits(chunk) * coefficients).sum(dim=1) for chunk in torch.split(candidates, 16384)])

        # Within each run of equal hashes, compare every later member with the run's first (lowest-position) one.
        order = torch.argsort(hashes, stable=True)
        sorted_hashes = hashes[order]
        repeat = torch.zeros(len(order), dtype=torch.bool, device=self.device)
        repeat[1:] = sorted_hashes[1:] == sorted_hashes[:-1]
        positions = torch.arange(len(order), device=self.device)
        run_start = torch.cummax(torch.where(repeat, 0, positions), dim=0).values
        later, first = order[repeat], order[run_start[repeat]]
        identical = torch.cat([
            (raw_bits(candidates[a]) == raw_bits(candidates[b])).all(dim=1)
            for a, b in zip(torch.split(later, 16384), torch.split(first, 16384))
        ]) if len(later) else torch.zeros(0, dtype=torch.bool, device=self.device)

        keep = torch.ones(len(candidates), dtype=torch.bool, device=self.device)
        keep[later[identical]] = False
        return torch.nonzero(keep).squeeze(1).cpu()

    def _filtered_replay_indices(self, replay_count):
        """Return replay indices from one validity-filtered candidate pool."""
        valid_indices = []
        valid_scene_coords = []
        valid_errors = []
        selection_start = time.perf_counter()
        chunk_size = 16384

        self.regressor.eval()
        with torch.no_grad():
            for chunk_start in range(0, self.options.training_buffer_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, self.options.training_buffer_size)
                features = self.training_buffer['features'][chunk_start:chunk_end]
                with autocast(enabled=self.options.use_half):
                    scene_coords = self.regressor.get_scene_coordinates(
                        features.unsqueeze(2).unsqueeze(3)
                    ).squeeze(3).squeeze(2).float()

                scene_coords_h = to_homogeneous(scene_coords.unsqueeze(2))
                camera_coords = torch.bmm(
                    self.training_buffer['gt_poses_inv'][chunk_start:chunk_end],
                    scene_coords_h,
                )
                projected = torch.bmm(
                    self.training_buffer['intrinsics'][chunk_start:chunk_end],
                    camera_coords,
                )
                depth = camera_coords[:, 2, 0]
                projected_px = projected[:, :2, 0] / projected[:, 2:3, 0].clamp_min(self.options.depth_min)
                reprojection_error = torch.norm(
                    projected_px - self.training_buffer['target_px'][chunk_start:chunk_end],
                    dim=1,
                    p=1,
                )
                valid = (
                    (depth >= self.options.depth_min) &
                    (depth <= self.options.depth_max) &
                    (reprojection_error <= self.options.replay_reprojection_error_px)
                )
                local_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
                if len(local_indices):
                    valid_indices.append((local_indices + chunk_start).cpu())
                    valid_scene_coords.append(scene_coords[local_indices].cpu())
                    valid_errors.append(reprojection_error[local_indices].cpu())

        self.regressor.train()
        if not valid_indices:
            raise RuntimeError("No replay candidates passed the reprojection filter")

        candidate_indices = torch.cat(valid_indices)
        candidate_coords = torch.cat(valid_scene_coords)
        if len(candidate_indices) < replay_count:
            raise RuntimeError(
                f"Only {len(candidate_indices)} replay candidates passed filtering; requested {replay_count}"
            )

        candidate_errors = torch.cat(valid_errors)
        pool_metadata = {}
        if self.drop_repeated_candidates:
            kept = self._first_copy_positions(candidate_indices)
            pool_metadata["repeated_candidates_dropped"] = len(candidate_indices) - len(kept)
            candidate_indices, candidate_coords, candidate_errors = (
                candidate_indices[kept], candidate_coords[kept], candidate_errors[kept])
            if len(candidate_indices) < replay_count:
                raise RuntimeError(
                    f"Only {len(candidate_indices)} distinct replay candidates remain; requested {replay_count}"
                )
        method = self.options.replay_selection
        method_metadata = {}
        if method == 'voxel_feature_dedup':
            selected_local, method_metadata = voxel_feature_dedup_sample(
                candidate_coords, self.training_buffer['features'], replay_count,
                self.options.replay_voxel_size_m, self.replay_generator,
                cosine_threshold=self.options.replay_dedup_cosine,
                pool_factor=self.options.replay_dedup_pool_factor,
                feature_indices=candidate_indices)
            occupied_voxels = None
        elif method in ("voxel", "voxel_quality"):
            selected_local, occupied_voxels = balanced_voxel_sample(
                candidate_coords, replay_count, self.options.replay_voxel_size_m,
                self.replay_generator,
                quality=candidate_errors if method == "voxel_quality" else None,
            )
        elif method == "voxel_weighted":
            selected_local, occupied_voxels = weighted_voxel_sample(
                candidate_coords, replay_count, self.options.replay_voxel_size_m,
                self.replay_generator, alpha=self.voxel_alpha)
            method_metadata = {"voxel_alpha": self.voxel_alpha}
        elif method == 'fps':
            selected_local = farthest_point_sample(
                candidate_coords, replay_count, self.replay_generator, device=self.device)
            occupied_voxels = None
        elif method == "quality":
            order = torch.randperm(len(candidate_indices), generator=self.replay_generator)
            selected_local = order[torch.argsort(candidate_errors[order], stable=True)[:replay_count]]
            occupied_voxels = None
        else:
            selected_local = torch.randperm(len(candidate_indices), generator=self.replay_generator)[:replay_count]
            occupied_voxels = None
        digest = hashlib.sha256()
        for tensor in (candidate_indices, candidate_coords, candidate_errors):
            digest.update(tensor.contiguous().numpy().tobytes())
        selected_coords = candidate_coords[selected_local]
        self.selection_diagnostics = {
            **pool_metadata,
            **method_metadata,
            "candidate_geometry_sha256": digest.hexdigest(),
            "selected_occupied_voxels": len(torch.unique(
                torch.floor(selected_coords / self.options.replay_voxel_size_m).long(), dim=0)),
            "selected_error_mean_px": candidate_errors[selected_local].mean().item(),
            "selection_seconds": time.perf_counter() - selection_start,
        }
        return candidate_indices[selected_local], len(candidate_indices), occupied_voxels

    def export_replay_buffer(self):
        """Export a deterministic fixed-size feature replay buffer without RGB images."""
        replay_count = min(self.options.replay_buffer_features, self.options.training_buffer_size)
        selection_metadata = {}
        if self.options.replay_selection != "random":
            replay_indices, valid_candidates, occupied_voxels = self._filtered_replay_indices(replay_count)
            selection_metadata = {
                "voxel_size_m": self.options.replay_voxel_size_m,
                "reprojection_error_px_l1": self.options.replay_reprojection_error_px,
                "valid_candidates": valid_candidates,
                "occupied_voxels": occupied_voxels,
                **self.selection_diagnostics,
            }
        else:
            replay_indices = torch.randperm(
                self.options.training_buffer_size,
                generator=self.replay_generator,
            )[:replay_count]

        replay_data = {
            key: value[replay_indices].detach().cpu()
            for key, value in self.training_buffer.items()
        }

        feature_encoding = self.options.replay_feature_encoding
        feature_scales = None
        if self.options.replay_feature_encoding == "int8":
            replay_data['features'], feature_scales = quantize_features_int8(replay_data['features'])
            feature_encoding = INT8_ENCODING
        elif self.options.replay_feature_encoding == 'fp32':
            replay_data['features'] = replay_data['features'].float()
            feature_encoding = 'fp32'
        else:
            replay_data['features'] = replay_data['features'].to(
                torch.float32 if feature_encoding == FP32_ENCODING else torch.float16)

        replay_payload = {
            "format_version": 2,
            "selection": self.options.replay_selection,
            "selection_metadata": selection_metadata,
            "feature_encoding": feature_encoding,
            "feature_scales": feature_scales,
            "seed": self.base_seed,
            "num_features": replay_count,
            "feature_dim": self.regressor.feature_dim,
            "data": replay_data,
        }
        self.options.export_replay_buffer.parent.mkdir(parents=True, exist_ok=True)
        torch.save(replay_payload, self.options.export_replay_buffer)
        replay_bytes = self.options.export_replay_buffer.stat().st_size
        self.replay_features_stored = replay_count
        self.replay_buffer_bytes = replay_bytes
        self.replay_selection = self.options.replay_selection
        self.replay_feature_encoding = feature_encoding
        _logger.info(
            "Exported %d %s replay features as %s (%.2fMB, no RGB) to: %s",
            replay_count,
            self.options.replay_selection,
            feature_encoding,
            replay_bytes / 1024 ** 2,
            self.options.export_replay_buffer,
        )

    def inject_replay_buffer(self):
        """Replace a fixed fraction of the current feature buffer with stored replay tuples."""
        replay_payload = torch.load(self.options.replay_buffer, map_location="cpu")
        if replay_payload.get("format_version") not in (1, 2):
            raise ValueError("Unsupported replay buffer format")
        replay_data = replay_payload.get("data", {})
        if set(replay_data) != set(self.training_buffer):
            raise ValueError("Replay buffer fields do not match the ACE training buffer")
        if replay_payload.get("feature_dim") != self.regressor.feature_dim:
            raise ValueError("Replay feature dimension does not match the encoder")

        stored_count = replay_payload.get("num_features", 0)
        if stored_count < 1 or any(len(value) != stored_count for value in replay_data.values()):
            raise ValueError("Replay buffer has inconsistent or empty tensors")

        feature_encoding = replay_payload.get("feature_encoding", FP16_ENCODING)
        if self.options.precision == 'fp32' and (
                feature_encoding != FP32_ENCODING or replay_data['features'].dtype != torch.float32):
            raise ValueError('FP32 protocol requires replay features extracted and stored as FP32')
        feature_scales = replay_payload.get("feature_scales")
        if feature_encoding == INT8_ENCODING:
            if feature_scales is None or len(feature_scales) != self.regressor.feature_dim:
                raise ValueError("INT8 replay buffer has invalid per-channel scales")
        elif feature_encoding not in (FP16_ENCODING, FP32_ENCODING):
            raise ValueError(f"Unsupported replay feature encoding: {feature_encoding}")

        inject_count = round(self.options.training_buffer_size * self.options.replay_ratio_percent / 100)
        write_start = self.options.training_buffer_size - inject_count
        chunk_size = 50000
        for chunk_start in range(0, inject_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, inject_count)
            source_indices = torch.randint(
                stored_count,
                (chunk_end - chunk_start,),
                generator=self.replay_generator,
            )
            target_slice = slice(write_start + chunk_start, write_start + chunk_end)
            for key in self.training_buffer:
                source = replay_data[key][source_indices]
                if key == 'features' and feature_encoding == INT8_ENCODING:
                    source = dequantize_features_int8(
                        source,
                        feature_scales,
                        dtype=self.training_buffer[key].dtype,
                    )
                self.training_buffer[key][target_slice] = source.to(self.device, non_blocking=False)

        self.replay_features_injected = inject_count
        self.replay_features_stored = stored_count
        self.replay_buffer_bytes = self.options.replay_buffer.stat().st_size
        self.replay_selection = replay_payload.get("selection", "uniform_random")
        self.replay_feature_encoding = feature_encoding
        _logger.info(
            "Injected %d replay features (%.2f%% of %d training entries) from a fixed %.2fMB buffer.",
            inject_count,
            self.options.replay_ratio_percent,
            self.options.training_buffer_size,
            self.replay_buffer_bytes / 1024 ** 2,
        )

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        if self.options.replay_batch_mode == "stratified" and self.replay_features_injected:
            batches = stratified_epoch_indices(
                self.options.training_buffer_size, self.replay_features_injected,
                self.options.batch_size, self.options.replay_ratio_percent,
                self.training_generator,
            )
        else:
            order = torch.randperm(self.options.training_buffer_size, generator=self.training_generator)
            stop = len(order) // self.options.batch_size * self.options.batch_size
            batches = (order[start:start + self.options.batch_size]
                       for start in range(0, stop, self.options.batch_size))
        for random_batch_indices in batches:
            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous(),
                self.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous()
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        # Upstream read optimizer._step_count, which newer PyTorch versions no longer set. The scaler skips a step
        # exactly when it finds inf/NaN gradients, and then it always lowers the scale.
        old_scale = self.scaler.get_scale() if self.scaler.is_enabled() else None

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        optimizer_stepped = old_scale is None or self.scaler.get_scale() >= old_scale
        if optimizer_stepped:
            self.optimizer_steps += 1

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if optimizer_stepped and self.optimizer_steps < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self, output_path=None):
        if output_path is None:
            output_path = self.options.output_map_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        for k, v in head_state_dict.items():
            # fp32 coordinates keep the head in fp32 too: fp16 would round the final layer and the center.
            keep_fp32 = self.options.precision == 'fp32' or self.fp32_coordinates
            head_state_dict[k] = v.float() if keep_fp32 else v.half()
        torch.save(head_state_dict, output_path)
        _logger.info(f"Saved trained head weights to: {output_path}")
