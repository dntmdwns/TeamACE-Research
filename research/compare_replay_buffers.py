#!/usr/bin/env python3
"""Verify that two replay payloads contain identical tensors and metadata."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu")
    right = torch.load(args.right, map_location="cpu")
    metadata_keys = (
        "format_version",
        "selection",
        "selection_metadata",
        "feature_encoding",
        "seed",
        "num_features",
        "feature_dim",
    )
    for key in metadata_keys:
        if left.get(key) != right.get(key):
            raise AssertionError(f"Metadata differs for {key}: {left.get(key)!r} != {right.get(key)!r}")

    left_scales = left.get("feature_scales")
    right_scales = right.get("feature_scales")
    if (left_scales is None) != (right_scales is None):
        raise AssertionError("Only one buffer contains feature scales")
    if left_scales is not None and not torch.equal(left_scales, right_scales):
        raise AssertionError("Feature scales differ")

    if set(left["data"]) != set(right["data"]):
        raise AssertionError("Replay data fields differ")
    for key in left["data"]:
        if not torch.equal(left["data"][key], right["data"][key]):
            raise AssertionError(f"Replay tensor differs: {key}")
    print("Replay buffers are identical.")


if __name__ == "__main__":
    main()
