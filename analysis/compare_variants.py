#!/usr/bin/env python3
"""Compare N variants across all shared instances.

Supports 2-way (fork vs official) and 3-way (fork vs official vs tree).

Usage:
    python compare_variants.py <dir1> <dir2> [dir3 ...] [--csv output.csv]

Example:
    python compare_variants.py \
        outputs/full-dev-opencode-mimo-v2-pro-free-c24/fork \
        outputs/full-dev-opencode-mimo-v2-pro-free-c24/official \
        outputs/full-dev-opencode-mimo-v2-pro-free-c24/tree \
        --csv comparison_3way.csv
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

# Lower is better for all metrics except patch_non_empty (higher is better).
METRICS = [
    "tokens_total",
    "tokens_input",
    "tokens_output",
    "tokens_cache_read",
    "tokens_cache_write",
    "duration_seconds",
    "time_to_first_edit_seconds",
    "steps",
    "tool_calls_total",
    "failed_tool_calls",
    "files_read_count",
    "files_edited_count",
    "dead_ends_count",
    "patch_non_empty",
]

HIGHER_IS_BETTER = {"patch_non_empty"}


def ensure_parsed(json_path: Path) -> dict:
    parsed_path = json_path.with_name(json_path.stem + "_parsed.json")
    if not parsed_path.exists():
        subprocess.run(
            [sys.executable, "analysis/parse_instance.py", str(json_path), "--quiet"],
            check=True,
        )
    with open(parsed_path) as f:
        return json.load(f)


def extract_metrics(parsed: dict) -> dict[str, float]:
    t = parsed["tokens"]
    return {
        "tokens_total": t["total"],
        "tokens_input": t["input"],
        "tokens_output": t["output"],
        "tokens_cache_read": t["cache_read"],
        "tokens_cache_write": t["cache_write"],
        "duration_seconds": parsed["duration_seconds"] or 0,
        "time_to_first_edit_seconds": parsed["time_to_first_edit_seconds"] or 0,
        "steps": parsed["steps"],
        "tool_calls_total": parsed["tool_calls"]["total"],
        "failed_tool_calls": parsed["failed_tool_calls"],
        "files_read_count": len(parsed["files_read"]),
        "files_edited_count": len(parsed["files_edited"]),
        "dead_ends_count": len(parsed["dead_ends"]),
        "patch_non_empty": 1 if parsed["patch_non_empty"] else 0,
    }


def instance_files(d: Path) -> dict[str, Path]:
    results = {}
    for p in sorted(d.glob("*.json")):
        if p.stem.endswith("_parsed") or p.stem.endswith("_tree"):
            continue
        results[p.stem] = p
    return results


def variant_name(d: Path) -> str:
    return d.name


def main():
    # Parse args: positional dirs, optional --csv
    dirs: list[Path] = []
    csv_path = Path("comparison.csv")
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--csv" and i + 1 < len(args):
            csv_path = Path(args[i + 1])
            i += 2
        elif args[i].startswith("-"):
            print(f"Unknown flag: {args[i]}", file=sys.stderr)
            sys.exit(1)
        else:
            dirs.append(Path(args[i]))
            i += 1

    if len(dirs) < 2:
        print(
            "Usage: python compare_variants.py <dir1> <dir2> [dir3 ...] [--csv output.csv]",
            file=sys.stderr,
        )
        sys.exit(1)

    variants = [variant_name(d) for d in dirs]
    variant_files: dict[str, dict[str, Path]] = {}
    for d, v in zip(dirs, variants):
        variant_files[v] = instance_files(d)

    # Find instances shared across ALL variants
    shared_sets = [set(variant_files[v].keys()) for v in variants]
    shared = sorted(shared_sets[0].intersection(*shared_sets[1:]))

    if not shared:
        print("No shared instances found across all directories.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing {len(variants)} variants: {', '.join(variants)}")
    print(f"Shared instances: {len(shared)}\n")

    # Collect metrics per variant per instance
    all_metrics: dict[str, dict[str, list[float]]] = {
        v: {m: [] for m in METRICS} for v in variants
    }
    wins: dict[str, dict[str, int]] = {
        v: {m: 0 for m in METRICS} for v in variants
    }

    rows: list[dict] = []

    for instance_id in shared:
        row: dict[str, object] = {"instance_id": instance_id}
        instance_metrics: dict[str, dict[str, float]] = {}

        for v in variants:
            parsed = ensure_parsed(variant_files[v][instance_id])
            metrics = extract_metrics(parsed)
            instance_metrics[v] = metrics
            for m in METRICS:
                row[f"{v}_{m}"] = metrics[m]
                all_metrics[v][m].append(metrics[m])

        # Determine winner for each metric on this instance
        for m in METRICS:
            values = {v: instance_metrics[v][m] for v in variants}
            if m in HIGHER_IS_BETTER:
                best_val = max(values.values())
            else:
                best_val = min(values.values())
            winners = [v for v, val in values.items() if val == best_val]
            if len(winners) == 1:
                wins[winners[0]][m] += 1

        rows.append(row)

    # Write CSV
    fieldnames = ["instance_id"]
    for m in METRICS:
        for v in variants:
            fieldnames.append(f"{v}_{m}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written to: {csv_path}\n")

    # Print summary table
    n = len(shared)
    # Header
    header = f"{'Metric':<32}"
    for v in variants:
        header += f" {v + ' Mean':>14}"
    header += "  Best"
    for v in variants:
        header += f" {v + ' Wins':>10}"
    print(header)
    print("-" * len(header))

    for m in METRICS:
        line = f"{m:<32}"
        means = {}
        for v in variants:
            mean = sum(all_metrics[v][m]) / n
            means[v] = mean
            line += f" {mean:>14.1f}"

        if m in HIGHER_IS_BETTER:
            best_v = max(means, key=lambda v: means[v])
        else:
            best_v = min(means, key=lambda v: means[v])

        # Mark best (or tie)
        best_val = means[best_v]
        tied = [v for v, val in means.items() if val == best_val]
        if len(tied) == len(variants):
            line += f"  {'tie':>6}"
        else:
            line += f"  {best_v:>6}"

        for v in variants:
            line += f" {wins[v][m]:>10}"

        print(line)

    # Overall
    print()
    for v in variants:
        total = sum(wins[v].values())
        print(f"{v}: {total} metric-instance wins")


if __name__ == "__main__":
    main()
