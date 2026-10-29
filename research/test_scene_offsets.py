#!/usr/bin/env python3
"""CPU tests for the shared-frame options: per-scene camera offsets (dataset.py --scene_offset) and fp32 scene
coordinates in the head's final layer (ace_network.Head.fp32_coordinates). Both are off by default."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ace_network import Head  # noqa: E402
from dataset import CamLocDataset  # noqa: E402


def random_pose(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    pose = np.eye(4)
    pose[:3, :3] = q * np.sign(np.linalg.det(q))
    pose[:3, 3] = rng.uniform(-2, 2, size=3)
    return pose


class SceneOffsetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "train"
        rng = np.random.default_rng(0)
        self.poses = [random_pose(rng) for _ in range(4)]
        for sub in ("rgb", "poses", "calibration"):
            (self.root / sub).mkdir(parents=True)
        import cv2
        for i, pose in enumerate(self.poses):
            cv2.imwrite(str(self.root / "rgb" / f"f{i}.png"), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
            np.savetxt(self.root / "poses" / f"f{i}.txt", pose)
            (self.root / "calibration" / f"f{i}.txt").write_text("525.0\n")

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, offset=None):
        return CamLocDataset(self.root, mode=0, image_height=48, augment=False, scene_offset=offset)

    def test_off_by_default_and_translation_only(self):
        native, shifted = self.dataset(), self.dataset([8.0, -1.5, 0.25])
        self.assertIsNone(native.scene_offset)
        for i, pose in enumerate(self.poses):
            p0, p1 = native._load_pose(i), shifted._load_pose(i)
            self.assertTrue(torch.equal(p0, torch.from_numpy(pose).float()))   # untouched without an offset
            self.assertTrue(torch.equal(p1[:3, :3], p0[:3, :3]))               # rotation unchanged
            self.assertTrue(torch.equal(p1[3], p0[3]))
            self.assertTrue(torch.allclose(p1[:3, 3] - p0[:3, 3], torch.tensor([8.0, -1.5, 0.25]), atol=1e-6))
        self.assertTrue(torch.allclose(shifted.mean_cam_center - native.mean_cam_center,
                                       torch.tensor([8.0, -1.5, 0.25]), atol=1e-6))

    def test_items_use_the_shifted_pose(self):
        native, shifted = self.dataset(), self.dataset([15.0, 0.0, 0.0])
        _, _, pose0, inv0, k0, _, _, _ = native[0]
        image, _, pose1, inv1, k1, _, _, _ = shifted[0]
        self.assertTrue(torch.equal(k0, k1))
        self.assertTrue(torch.allclose(pose1[:3, 3], pose0[:3, 3] + torch.tensor([15.0, 0, 0]), atol=1e-6))
        # The inverse pose maps the shifted camera center to the origin.
        center = torch.cat([pose1[:3, 3], torch.ones(1)])
        self.assertTrue(torch.allclose(inv1 @ center, torch.tensor([0, 0, 0, 1.0]), atol=1e-5))


class Fp32CoordinatesTest(unittest.TestCase):
    def head(self, mean):
        torch.manual_seed(0)
        head = Head(torch.tensor(mean), num_head_blocks=1, use_homogeneous=True)
        with torch.no_grad():   # coordinates about 15 m from the center, as for a scene placed 15 m away
            head.fc3.weight *= 0.01
            head.fc3.bias[:] = torch.tensor([15.0, -3.0, 2.0, 0.0])   # softplus(0) + max_inv_scale = 1
        return head.eval()

    def reference(self, head, features):
        """The upstream forward in float64."""
        h = head.double()
        x = torch.relu(h.res3_conv1(features)); x = torch.relu(h.res3_conv2(x)); x = torch.relu(h.res3_conv3(x))
        res = h.head_skip(features) + x
        for block in h.res_blocks:
            x = torch.relu(block[0](res)); x = torch.relu(block[1](x)); x = torch.relu(block[2](x))
            res = res + x
        sc = h.fc3(torch.relu(h.fc2(torch.relu(h.fc1(res)))))
        w = (torch.nn.functional.softplus(sc[:, 3:4], beta=h.h_beta.item()) + h.max_inv_scale).clamp(
            max=h.min_inv_scale.item())
        return sc[:, :3] / w + h.mean

    def test_off_by_default(self):
        self.assertFalse(Head.fp32_coordinates)
        self.assertFalse(self.head([0.0, 0.0, 0.0]).fp32_coordinates)

    def test_matches_fp64_and_upstream_fp32(self):
        head = self.head([0.4, -0.7, 0.3])
        features = torch.relu(torch.randn(2, 512, 4, 4, generator=torch.Generator().manual_seed(1)))
        with torch.no_grad():
            upstream = head(features)
            head.fp32_coordinates = True
            fp32 = head(features)
            reference = self.reference(self.head([0.4, -0.7, 0.3]), features.double())
        self.assertEqual(fp32.dtype, torch.float32)
        self.assertGreater(reference[:, 0].mean().item(), 10)   # really ~15 m from the center
        self.assertLess((fp32.double() - reference).abs().max().item(), 1e-5)
        self.assertLess((fp32 - upstream).abs().max().item(), 1e-5)   # same math as upstream in plain fp32

    def test_trains(self):
        head = self.head([0.0, 0.0, 0.0])
        head.fp32_coordinates = True
        head.train()
        features = torch.relu(torch.randn(1, 512, 2, 2))
        head(features).sum().backward()
        self.assertIsNotNone(head.fc3.weight.grad)
        self.assertGreater(head.fc3.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.fc1.weight.grad.abs().sum().item(), 0)

    def test_requires_tf32_off_on_cuda_only(self):
        head = self.head([0.0, 0.0, 0.0])
        head.fp32_coordinates = True
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True   # ignored on the CPU, where there is no TF32
            with torch.no_grad():
                head(torch.zeros(1, 512, 1, 1))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


if __name__ == "__main__":
    unittest.main()
