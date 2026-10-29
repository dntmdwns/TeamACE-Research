#!/usr/bin/env python3
"""Aggregate full-test replay results across explicitly named seeds."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_seed_result(value):
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected SEED=PATH")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from error
    return seed, Path(path_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_result", action="append", type=parse_seed_result, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_ratio = defaultdict(list)
    for seed, path in args.seed_result:
        with path.open() as input_file:
            payload = json.load(input_file)
        for row in payload["results"]:
            rows_by_ratio[float(row["replay_ratio_percent"])].append({
                "seed": seed,
                "chess": row["chess_5cm_5deg_percent"],
                "fire": row["fire_5cm_5deg_percent"],
                "average": row["average_5cm_5deg_percent"],
            })

    results = []
    for ratio in sorted(rows_by_ratio):
        seed_rows = sorted(rows_by_ratio[ratio], key=lambda row: row["seed"])
        aggregate = {}
        for metric in ("chess", "fire", "average"):
            values = [row[metric] for row in seed_rows]
            aggregate[f"{metric}_mean_percent"] = statistics.mean(values)
            aggregate[f"{metric}_sample_std_percent"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        results.append({
            "replay_ratio_percent": ratio,
            "num_seeds": len(seed_rows),
            "seeds": seed_rows,
            **aggregate,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump({"results": results}, output_file, indent=2)
        output_file.write("\n")
    print(f"Saved multi-seed summary to: {args.output}")


if __name__ == "__main__":
    main()
