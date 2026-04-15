"""Aggregate per-variant observability stats from a run.

Reads:
  outputs/<variant>/<instance>.json      (summary: duration, input, output, cache_read)
  outputs/<variant>/logs/<instance>.log  (raw NDJSON event stream)

Prints a comparison table across variants covering:
  - completion rate (how many instances produced non-empty output)
  - patch produced rate (non-zero patch_chars)
  - latency (mean / p50 / p90)
  - aggregate tokens (input, output, cache_read)
  - tool calls split main vs subagent
  - session count distribution (1, 2, 3-5, 6+)
  - cache hit/miss verdict for subagent first-step calls
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_ndjson(path: Path) -> list[dict]:
    events = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except FileNotFoundError:
        return []
    return events


def analyze_instance(summary_path: Path, log_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    events = parse_ndjson(log_path)

    session_order: list[str] = []
    tool_calls_per_session: dict[str, Counter] = defaultdict(Counter)
    first_step: dict[str, dict] = {}
    step_count: dict[str, int] = defaultdict(int)

    for e in events:
        sid = e.get("sessionID") or e.get("part", {}).get("sessionID")
        if not sid:
            continue
        if sid not in session_order:
            session_order.append(sid)
        etype = e.get("type")
        part = e.get("part", {}) or {}
        if etype == "tool_use":
            tool = part.get("tool", "?")
            tool_calls_per_session[sid][tool] += 1
        elif etype == "step_finish":
            step_count[sid] += 1
            if sid not in first_step:
                tokens = part.get("tokens", {}) or {}
                cache = tokens.get("cache", {}) or {}
                first_step[sid] = {
                    "input": tokens.get("input", 0),
                    "cache_read": cache.get("read", 0),
                }

    main_sid = session_order[0] if session_order else None
    subagent_sids = session_order[1:]

    main_tools = sum(tool_calls_per_session[main_sid].values()) if main_sid else 0
    subagent_tools = sum(
        sum(tool_calls_per_session[s].values()) for s in subagent_sids
    )

    cache_verdicts = []
    for sid in subagent_sids:
        fst = first_step.get(sid)
        if not fst or fst["input"] == 0:
            cache_verdicts.append("no_data")
            continue
        ratio = fst["cache_read"] / fst["input"]
        if ratio >= 0.8:
            cache_verdicts.append("cache_hit")
        elif ratio <= 0.2:
            cache_verdicts.append("cache_miss")
        else:
            cache_verdicts.append("partial")

    return {
        "instance_id": summary.get("instance_id"),
        "duration": summary.get("duration", 0),
        "input_tokens": summary.get("input", 0),
        "output_tokens": summary.get("output", 0),
        "cache_read": summary.get("cache_read", 0),
        "patch_chars": len(summary.get("model_patch") or ""),
        "session_count": len(session_order),
        "main_tool_calls": main_tools,
        "subagent_tool_calls": subagent_tools,
        "cache_verdicts": cache_verdicts,
        "log_empty": len(events) == 0,
    }


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def summarize_variant(variant_dir: Path) -> dict:
    summary_files = sorted(variant_dir.glob("*.json"))
    log_dir = variant_dir / "logs"
    instances = []
    for sp in summary_files:
        lp = log_dir / (sp.stem + ".log")
        try:
            instances.append(analyze_instance(sp, lp))
        except Exception as ex:
            print(f"  skip {sp.name}: {ex}")

    total = len(instances)
    if total == 0:
        return {"variant": variant_dir.name, "total": 0}

    completed = [i for i in instances if not i["log_empty"]]
    with_patch = [i for i in instances if i["patch_chars"] > 0]

    durations = [i["duration"] for i in completed]
    session_counts = [i["session_count"] for i in completed]
    all_verdicts = [v for i in completed for v in i["cache_verdicts"]]
    session_bucket = Counter()
    for c in session_counts:
        if c <= 1: session_bucket["1"] += 1
        elif c == 2: session_bucket["2"] += 1
        elif c <= 5: session_bucket["3-5"] += 1
        else: session_bucket["6+"] += 1

    return {
        "variant": variant_dir.name,
        "total": total,
        "completed": len(completed),
        "with_patch": len(with_patch),
        "dur_mean": round(statistics.mean(durations), 1) if durations else 0,
        "dur_p50": round(percentile(durations, 0.5), 1),
        "dur_p90": round(percentile(durations, 0.9), 1),
        "input_mean": round(statistics.mean([i["input_tokens"] for i in completed]), 0) if completed else 0,
        "output_mean": round(statistics.mean([i["output_tokens"] for i in completed]), 0) if completed else 0,
        "cache_read_mean": round(statistics.mean([i["cache_read"] for i in completed]), 0) if completed else 0,
        "main_tools_mean": round(statistics.mean([i["main_tool_calls"] for i in completed]), 1) if completed else 0,
        "sub_tools_mean": round(statistics.mean([i["subagent_tool_calls"] for i in completed]), 1) if completed else 0,
        "session_dist": dict(session_bucket),
        "cache_verdicts": dict(Counter(all_verdicts)),
    }


def print_table(summaries: list[dict]) -> None:
    if not summaries:
        print("no data"); return

    rows = [
        ("total",            lambda s: f"{s.get('total', 0)}"),
        ("completed",        lambda s: f"{s.get('completed', 0)}/{s.get('total', 0)}"),
        ("with_patch",       lambda s: f"{s.get('with_patch', 0)}/{s.get('total', 0)}"),
        ("duration mean",    lambda s: f"{s.get('dur_mean', 0)}s"),
        ("duration p50",     lambda s: f"{s.get('dur_p50', 0)}s"),
        ("duration p90",     lambda s: f"{s.get('dur_p90', 0)}s"),
        ("input tokens",     lambda s: f"{int(s.get('input_mean', 0)):,}"),
        ("output tokens",    lambda s: f"{int(s.get('output_mean', 0)):,}"),
        ("cache_read",       lambda s: f"{int(s.get('cache_read_mean', 0)):,}"),
        ("main tool calls",  lambda s: f"{s.get('main_tools_mean', 0)}"),
        ("subagent tool calls", lambda s: f"{s.get('sub_tools_mean', 0)}"),
        ("session dist",     lambda s: " ".join(f"{k}:{v}" for k, v in sorted(s.get('session_dist', {}).items()))),
        ("cache verdicts",   lambda s: " ".join(f"{k}:{v}" for k, v in s.get('cache_verdicts', {}).items())),
    ]

    col_width = 24
    variants = [s["variant"] for s in summaries]
    print()
    print(f"{'METRIC':<22}" + "".join(f"{v:<{col_width}}" for v in variants))
    print("-" * (22 + col_width * len(variants)))
    for name, getter in rows:
        print(f"{name:<22}" + "".join(f"{getter(s):<{col_width}}" for s in summaries))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    args = p.parse_args()

    outputs = Path(args.outputs_dir)
    variant_dirs = [d for d in sorted(outputs.iterdir()) if d.is_dir() and d.name != "logs"]

    summaries = [summarize_variant(d) for d in variant_dirs]
    print_table(summaries)


if __name__ == "__main__":
    main()
