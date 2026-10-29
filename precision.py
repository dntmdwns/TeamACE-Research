"""Explicit opt-in FP32 protocol; legacy ACE defaults remain unchanged."""
import torch


def configure_precision(profile):
    if profile == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def precision_metadata(profile, use_half):
    return {
        'profile': profile,
        'amp_enabled': bool(use_half),
        'feature_buffer_dtype': 'float16' if use_half else 'float32',
        'checkpoint_dtype': 'float32' if profile == 'fp32' else 'float16',
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }
