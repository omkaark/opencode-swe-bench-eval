import argparse
import json
import statistics
from pathlib import Path

import yaml
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, RUN_EVALUATION_LOG_DIR
from swebench.harness.run_evaluation import main as run_evaluation


DATASET = "princeton-nlp/SWE-bench_Lite"


def load_artifacts(output_dir: Path) -> dict[str, list[dict]]:
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


def patch_harness_install_commands() -> None:
    fallback_map = {
        "python -m pip install -e .": "python -m pip install -e . || python -m pip install .",
        "python -m pip install -e .[all]": "python -m pip install -e .[all] || python -m pip install .[all]",
        "python -m pip install -e '.[dev]'": "python -m pip install -e '.[dev]' || python -m pip install '.[dev]'",
        'python -m pip install -e ".[dev]"': 'python -m pip install -e ".[dev]" || python -m pip install ".[dev]"',
    }

    def rewrite_install(install):
        if isinstance(install, str):
            return fallback_map.get(install, install)
        if isinstance(install, list):
            return [fallback_map.get(item, item) for item in install]
        return install

    for version_specs in MAP_REPO_VERSION_TO_SPECS.values():
        for spec in version_specs.values():
            spec["install"] = rewrite_install(spec.get("install"))


def run_harness(output_dir: Path, variant: str, split: str) -> dict[str, bool]:
    predictions_path = output_dir / f"predictions_{variant}.jsonl"
    if not predictions_path.exists():
        print(f"No predictions file for {variant}")
        return {}

    instance_ids = []
    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            if pred.get("model_patch"):
                instance_ids.append(pred["instance_id"])

    if not instance_ids:
        print(f"No patches to evaluate for {variant}")
        return {}

    patch_harness_install_commands()

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
    results = {}
    log_dir = Path(RUN_EVALUATION_LOG_DIR) / variant
    if not log_dir.exists():
        return results

    for report_file in log_dir.rglob("report.json"):
        try:
            with open(report_file) as f:
                report = json.load(f)
            instance_id = report_file.parent.name
            inner = report.get(instance_id, {})
            results[instance_id] = inner.get("resolved", False)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0, "p50": 0, "p90": 0}
    return {
        "avg": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
    }


def count_tool_calls(log_content: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in log_content.splitlines():
        try:
            event = json.loads(line)
            if event.get("type") == "tool_use":
                tool = event.get("part", {}).get("tool", "unknown")
                counts[tool] = counts.get(tool, 0) + 1
        except json.JSONDecodeError:
            continue
    return counts


def load_tool_calls(output_dir: Path, variant: str, artifacts: list[dict]) -> dict[str, dict[str, int]]:
    results = {}
    logs_dir = output_dir / variant / "logs"
    if not logs_dir.exists():
        return results

    for artifact in artifacts:
        instance_id = artifact["instance_id"]
        safe_name = instance_id.replace("/", "__")
        log_path = logs_dir / f"{safe_name}.log"
        if log_path.exists():
            results[instance_id] = count_tool_calls(log_path.read_text())
    return results


def aggregate_tool_calls(tool_calls_by_instance: dict[str, dict[str, int]]) -> dict[str, float]:
    if not tool_calls_by_instance:
        return {}

    all_tools: set[str] = set()
    for counts in tool_calls_by_instance.values():
        all_tools.update(counts.keys())

    averages = {}
    n = len(tool_calls_by_instance)
    for tool in sorted(all_tools):
        total = sum(counts.get(tool, 0) for counts in tool_calls_by_instance.values())
        averages[tool] = round(total / n, 2)
    return averages


def analyze_variant(artifacts: list[dict], harness_results: dict[str, bool]) -> dict:
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


def print_table(stats_by_variant: dict[str, dict], tool_calls_by_variant: dict[str, dict[str, float]]) -> None:
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

    # Tool calls table
    if tool_calls_by_variant:
        all_tools: set[str] = set()
        for tool_avgs in tool_calls_by_variant.values():
            all_tools.update(tool_avgs.keys())

        if all_tools:
            print("\n" + "=" * 80)
            print("TOOL CALLS (avg per instance)")
            print("=" * 80)

            tool_col_width = max(len(t) for t in all_tools) + 2
            header = f"{'Tool':<{tool_col_width}}"
            for variant in variants:
                header += f" {variant:>12}"
            print(f"\n{header}")
            print("-" * (tool_col_width + 13 * len(variants)))

            for tool in sorted(all_tools):
                row = f"{tool:<{tool_col_width}}"
                for variant in variants:
                    avg = tool_calls_by_variant.get(variant, {}).get(tool, 0)
                    row += f" {avg:>12}"
                print(row)

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
            print("No cached harness results found. Use --run-harness to run evaluation.\n")

    stats_by_variant = {
        variant: analyze_variant(artifacts, harness_results_by_variant.get(variant, {}))
        for variant, artifacts in artifacts_by_variant.items()
    }

    tool_calls_by_variant = {}
    for variant, artifacts in artifacts_by_variant.items():
        tool_calls = load_tool_calls(output_dir, variant, artifacts)
        tool_calls_by_variant[variant] = aggregate_tool_calls(tool_calls)

    print_table(stats_by_variant, tool_calls_by_variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file")
    parser.add_argument("--run-harness", action="store_true", help="Run SWE-bench harness evaluation")
    args = parser.parse_args()
    main(args.config, args.run_harness)
