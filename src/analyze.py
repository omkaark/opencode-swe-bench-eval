import argparse
import json
import statistics
from pathlib import Path

import yaml
from swebench.harness.run_evaluation import main as run_evaluation


DATASET = "princeton-nlp/SWE-bench_Lite"


def load_artifacts(output_dir: Path) -> dict[str, list[dict]]:
    """Load all artifact JSON files grouped by variant."""
    results = {}
    for variant_dir in output_dir.iterdir():
        if not variant_dir.is_dir() or variant_dir.name == "logs":
            continue
        variant = variant_dir.name
        artifacts = []
        for json_file in variant_dir.glob("*.json"):
            with open(json_file) as f:
                artifacts.append(json.load(f))
        if artifacts:
            results[variant] = artifacts
    return results


def run_harness(output_dir: Path, variant: str, split: str) -> dict[str, str]:
    """Run SWE-bench harness and return instance_id -> resolution status."""
    predictions_path = output_dir / f"predictions_{variant}.jsonl"
    if not predictions_path.exists():
        print(f"No predictions file for {variant}")
        return {}

    # Extract instance IDs with non-empty patches
    instance_ids = []
    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            if pred.get("model_patch"):
                instance_ids.append(pred["instance_id"])

    if not instance_ids:
        print(f"No patches to evaluate for {variant}")
        return {}

    print(f"Running harness for {variant} ({len(instance_ids)} instances)...")
    run_evaluation(
        dataset_name=DATASET,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=8,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=variant,
        timeout=1800,
        namespace=None,
        rewrite_reports=False,
        modal=True,
    )

    return load_harness_results(variant)


def load_harness_results(variant: str) -> dict[str, bool]:
    """Load harness results from report files."""
    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

    results = {}
    log_dir = Path(RUN_EVALUATION_LOG_DIR) / variant
    if not log_dir.exists():
        return results

    for report_file in log_dir.rglob("report.json"):
        try:
            with open(report_file) as f:
                report = json.load(f)
            instance_id = report_file.parent.name
            # Report is nested: {instance_id: {resolved: bool, ...}}
            inner = report.get(instance_id, {})
            results[instance_id] = inner.get("resolved", False)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile (0-100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute avg, p50, p90 for a list of values."""
    if not values:
        return {"avg": 0, "p50": 0, "p90": 0}
    return {
        "avg": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
    }


def analyze_variant(artifacts: list[dict], harness_results: dict[str, bool]) -> dict:
    """Analyze stats for a single variant."""
    durations = [a["duration"] for a in artifacts if a.get("duration") is not None]
    inputs = [a["input"] for a in artifacts if a.get("input") is not None]
    outputs = [a["output"] for a in artifacts if a.get("output") is not None]
    cache_reads = [a["cache_read"] for a in artifacts if a.get("cache_read") is not None]
    cache_writes = [a["cache_write"] for a in artifacts if a.get("cache_write") is not None]
    costs = [a["cost"] for a in artifacts if a.get("cost") is not None]
    patches = [len(a.get("model_patch", "")) for a in artifacts]

    patch_count = sum(1 for a in artifacts if a.get("model_patch"))
    resolved_count = sum(1 for a in artifacts if harness_results.get(a["instance_id"]) is True)

    return {
        "count": len(artifacts),
        "patch_count": patch_count,
        "resolved": resolved_count,
        "resolve_rate": round(resolved_count / len(artifacts) * 100, 1) if artifacts else 0,
        "duration": compute_stats(durations),
        "input_tokens": compute_stats(inputs),
        "output_tokens": compute_stats(outputs),
        "cache_read": compute_stats(cache_reads),
        "cache_write": compute_stats(cache_writes),
        "cost": compute_stats(costs),
        "patch_chars": compute_stats(patches),
    }


def print_table(stats_by_variant: dict[str, dict]) -> None:
    """Print a formatted table of stats."""
    if not stats_by_variant:
        print("No data found.")
        return

    variants = list(stats_by_variant.keys())
    col_width = max(len(v) for v in variants) + 2

    print("\n" + "=" * 80)
    print("VARIANT ANALYSIS")
    print("=" * 80)

    # Resolution summary
    print(f"\n{'Variant':<{col_width}} {'Resolved':>12} {'Rate':>10}")
    print("-" * (col_width + 24))
    for variant, stats in stats_by_variant.items():
        resolved_str = f"{stats['resolved']}/{stats['count']}"
        print(f"{variant:<{col_width}} {resolved_str:>12} {stats['resolve_rate']:>9}%")

    # Metrics tables
    metrics = [
        ("Duration (s)", "duration"),
        ("Input Tokens", "input_tokens"),
        ("Output Tokens", "output_tokens"),
        ("Cache Read", "cache_read"),
        ("Cache Write", "cache_write"),
        ("Cost ($)", "cost"),
        ("Patch Chars", "patch_chars"),
    ]

    for metric_name, metric_key in metrics:
        print(f"\n{metric_name}")
        print(f"{'Variant':<{col_width}} {'Avg':>12} {'P50':>12} {'P90':>12}")
        print("-" * (col_width + 38))
        for variant, stats in stats_by_variant.items():
            values = stats[metric_key]
            print(f"{variant:<{col_width}} {values['avg']:>12} {values['p50']:>12} {values['p90']:>12}")

    print()


def main(config_path: Path, run_harness_eval: bool) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output_dir", "outputs")).resolve()
    split = config.get("split", "dev")

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    artifacts_by_variant = load_artifacts(output_dir)

    if not artifacts_by_variant:
        print("No artifacts found.")
        return

    harness_results_by_variant = {}
    for variant in artifacts_by_variant:
        if run_harness_eval:
            harness_results_by_variant[variant] = run_harness(output_dir, variant, split)
        else:
            harness_results_by_variant[variant] = load_harness_results(variant)

    if not run_harness_eval:
        has_results = any(harness_results_by_variant.values())
        if not has_results:
            print("No cached harness results found. Use --run-harness to run evaluation.")
            return

    stats_by_variant = {
        variant: analyze_variant(artifacts, harness_results_by_variant.get(variant, {}))
        for variant, artifacts in artifacts_by_variant.items()
    }

    print_table(stats_by_variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--run-harness", action="store_true", help="Run SWE-bench harness evaluation")
    args = parser.parse_args()
    main(args.config, args.run_harness)
