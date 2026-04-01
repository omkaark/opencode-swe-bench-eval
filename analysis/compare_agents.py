#!/usr/bin/env python3
"""Cross-agent comparison across all 5 agents on SWE-bench Lite dev.

Produces per-instance deltas, averages, outliers,
token/tool-call breakdowns, and resolve rates.
"""

import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent

# --- agent output directories ---
AGENT_DIRS = {
    "official": BASE / "outputs" / "full-dev-opencode-mimo-v2-pro-free-c24" / "official",
    "fork":     BASE / "outputs" / "full-dev-opencode-mimo-v2-pro-free-c24" / "fork",
    "tree":     BASE / "outputs" / "tree",
    "claude":   BASE / "outputs" / "claude",
    "codex":    BASE / "outputs" / "codex",
}

# --- eval result files ---
EVAL_FILES = {
    "official": BASE / "outputs" / "eval-full-dev-opencode-mimo-v2-pro-free-c24" / "opencode-official.full_dev_opencode_mimo_v2_pro_free_c24_official.json",
    "fork":     BASE / "outputs" / "eval-full-dev-opencode-mimo-v2-pro-free-c24" / "opencode-fork.full_dev_opencode_mimo_v2_pro_free_c24_fork.json",
    "claude":   BASE / "opencode-claude.opencode-claude.json",
    "codex":    BASE / "opencode-codex.codex.json",
}
EVAL_DIRS = {
    "official": BASE / "logs" / "run_evaluation" / "full_dev_opencode_mimo_v2_pro_free_c24_official" / "opencode-official",
    "fork":     BASE / "logs" / "run_evaluation" / "full_dev_opencode_mimo_v2_pro_free_c24_fork" / "opencode-fork",
    "tree":     BASE / "logs" / "run_evaluation" / "tree" / "opencode-tree",
    "claude":   BASE / "logs" / "run_evaluation" / "opencode-claude" / "opencode-claude",
    "codex":    BASE / "logs" / "run_evaluation" / "codex" / "opencode-codex",
}


def load_resolved_ids(agent: str) -> set[str]:
    """Load the set of resolved instance IDs for an agent."""
    # Prefer per-instance report dirs (they survive partial reruns)
    if agent in EVAL_DIRS and EVAL_DIRS[agent].exists():
        resolved = set()
        for d in EVAL_DIRS[agent].iterdir():
            report = d / "report.json"
            if report.exists():
                r = json.loads(report.read_text())
                if r.get(d.name, {}).get("resolved", False):
                    resolved.add(d.name)
        return resolved

    return set()


def parse_opencode_ndjson(output: str) -> dict:
    """Extract tool calls and tokens from OpenCode NDJSON output."""
    tool_calls = 0
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    steps = 0

    for line in output.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        if etype == "tool_use":
            tool_calls += 1
        elif etype == "step_start":
            steps += 1
        elif etype == "step_finish":
            tokens = event.get("part", {}).get("tokens", {})
            total_tokens += tokens.get("total", 0)
            input_tokens += tokens.get("input", 0)
            output_tokens += tokens.get("output", 0)
            cache = tokens.get("cache", {})
            cache_read += cache.get("read", 0)
            cache_write += cache.get("write", 0)

    return {
        "tool_calls": tool_calls,
        "steps": steps,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        "cache_write": cache_write,
    }


def parse_claude_output(cli_output: str) -> dict:
    """Extract metrics from Claude Code JSON output."""
    try:
        data = json.loads(cli_output)
    except json.JSONDecodeError:
        return {"tool_calls": 0, "steps": 0, "total_tokens": 0,
                "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0}

    usage = data.get("usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cr = usage.get("cache_read_input_tokens", 0)
    cw = usage.get("cache_creation_input_tokens", 0)

    return {
        "tool_calls": data.get("num_turns", 0),
        "steps": data.get("num_turns", 0),
        "total_tokens": inp + out + cr + cw,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read": cr,
        "cache_write": cw,
    }


def parse_codex_output(cli_output: str) -> dict:
    """Extract metrics from Codex NDJSON output."""
    tool_calls = 0
    turns = 0

    for line in cli_output.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        if etype == "item.completed":
            item_type = event.get("item", {}).get("type", "")
            if item_type == "command_execution":
                tool_calls += 1
        elif etype == "turn.completed":
            turns += 1

    return {
        "tool_calls": tool_calls,
        "steps": turns,
        "total_tokens": 0,  # codex doesn't report tokens in output
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
    }


def load_agent_data(agent: str) -> dict[str, dict]:
    """Load all instance artifacts for an agent, return {instance_id: metrics}."""
    d = AGENT_DIRS[agent]
    if not d.exists():
        return {}

    results = {}
    for p in sorted(d.glob("*.json")):
        if p.stem.endswith("_parsed") or p.stem.endswith("_tree"):
            continue
        try:
            artifact = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue

        instance_id = artifact.get("instance_id", p.stem)
        duration = artifact.get("opencode_duration_seconds", 0) or 0
        patch = artifact.get("model_patch", "")

        # Parse agent-specific output
        if agent in ("official", "fork", "tree"):
            parsed = parse_opencode_ndjson(artifact.get("opencode_output", ""))
        elif agent == "claude":
            parsed = parse_claude_output(artifact.get("cli_output", ""))
        elif agent == "codex":
            parsed = parse_codex_output(artifact.get("cli_output", ""))
        else:
            parsed = {"tool_calls": 0, "steps": 0, "total_tokens": 0,
                       "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0}

        results[instance_id] = {
            "duration": round(duration, 2),
            "patch_len": len(patch),
            "has_patch": bool(patch.strip()),
            **parsed,
        }

    return results


def main():
    agents = ["official", "fork", "tree", "claude", "codex"]

    # Load data
    agent_data: dict[str, dict[str, dict]] = {}
    resolved: dict[str, set[str]] = {}
    for agent in agents:
        agent_data[agent] = load_agent_data(agent)
        resolved[agent] = load_resolved_ids(agent)

    # Find shared instances (present in all agents)
    all_ids = [set(agent_data[a].keys()) for a in agents]
    shared = sorted(all_ids[0].intersection(*all_ids[1:]))

    # Also find the union for resolve comparison
    union = sorted(all_ids[0].union(*all_ids[1:]))

    print("=" * 80)
    print("CROSS-AGENT COMPARISON — SWE-bench Lite Dev")
    print("=" * 80)
    print()

    # --- 1. resolve rates ---
    print("1. RESOLVE RATES")
    print("-" * 60)
    for agent in agents:
        n_instances = len(agent_data[agent])
        n_resolved = len(resolved[agent])
        pct = (n_resolved / n_instances * 100) if n_instances else 0
        print(f"  {agent:>10}: {n_resolved}/{n_instances} ({pct:.0f}%)")
    print()

    # --- 2. per-instance resolve matrix ---
    print("2. PER-INSTANCE RESOLVE MATRIX")
    print("-" * 80)
    header = f"  {'Instance':<40}"
    for a in agents:
        header += f" {a:>8}"
    print(header)
    print("  " + "-" * (40 + 9 * len(agents)))

    for iid in union:
        short = iid.split("__")[-1] if "__" in iid else iid
        line = f"  {short:<40}"
        for a in agents:
            if iid not in agent_data[a]:
                line += f" {'—':>8}"
            elif iid in resolved[a]:
                line += f" {'PASS':>8}"
            else:
                line += f" {'FAIL':>8}"
        print(line)
    print()

    # --- 3. unique solves ---
    print("3. UNIQUE SOLVES (solved by only one agent)")
    print("-" * 60)
    found_unique = False
    for agent in agents:
        unique = resolved[agent] - set().union(*(resolved[a] for a in agents if a != agent))
        if unique:
            found_unique = True
            for iid in sorted(unique):
                print(f"  {agent:>10} uniquely solved: {iid}")
    if not found_unique:
        print("  (none)")
    print()

    # --- 4. duration comparison on shared instances ---
    if shared:
        print(f"4. DURATION COMPARISON ({len(shared)} shared instances)")
        print("-" * 80)

        # Per-instance table
        header = f"  {'Instance':<40}"
        for a in agents:
            header += f" {a:>8}"
        print(header)
        print("  " + "-" * (40 + 9 * len(agents)))

        for iid in shared:
            short = iid.split("__")[-1] if "__" in iid else iid
            line = f"  {short:<40}"
            for a in agents:
                dur = agent_data[a][iid]["duration"]
                line += f" {dur:>7.0f}s"
            print(line)

        # Averages
        print("  " + "-" * (40 + 9 * len(agents)))
        line = f"  {'MEAN':<40}"
        for a in agents:
            mean_dur = statistics.mean(agent_data[a][iid]["duration"] for iid in shared)
            line += f" {mean_dur:>7.0f}s"
        print(line)
        line = f"  {'MEDIAN':<40}"
        for a in agents:
            med_dur = statistics.median(agent_data[a][iid]["duration"] for iid in shared)
            line += f" {med_dur:>7.0f}s"
        print(line)
        print()

        # --- 5. pairwise time deltas ---
        print("5. PAIRWISE TIME DELTAS (on shared instances)")
        print("-" * 60)

        baseline = "official"
        for agent in agents:
            if agent == baseline:
                continue
            deltas = []
            for iid in shared:
                d = agent_data[agent][iid]["duration"] - agent_data[baseline][iid]["duration"]
                deltas.append((iid, d))

            avg_delta = statistics.mean(d for _, d in deltas)
            med_delta = statistics.median(d for _, d in deltas)
            faster_count = sum(1 for _, d in deltas if d < 0)
            slower_count = sum(1 for _, d in deltas if d > 0)

            deltas_sorted = sorted(deltas, key=lambda x: x[1])
            fastest = deltas_sorted[0]
            slowest = deltas_sorted[-1]

            sign = "+" if avg_delta > 0 else ""
            print(f"  {agent} vs {baseline}:")
            print(f"    - Average delta: {sign}{avg_delta:.1f}s ({agent} {'slower' if avg_delta > 0 else 'faster'})")
            print(f"    - Median delta: {'+' if med_delta > 0 else ''}{med_delta:.1f}s")
            print(f"    - {agent} was faster on {faster_count}/{len(shared)}, slower on {slower_count}/{len(shared)}")
            print(f"    - Biggest gap favoring {agent}: {fastest[0].split('__')[-1]} ({abs(fastest[1]):.1f}s faster)")
            print(f"    - Biggest gap against {agent}: {slowest[0].split('__')[-1]} ({slowest[1]:.1f}s slower)")
            print()

    # --- 6. tool calls & tokens on shared instances ---
    if shared:
        print(f"6. TOOL CALLS & TOKENS ({len(shared)} shared instances)")
        print("-" * 80)

        header = f"  {'Metric':<25}"
        for a in agents:
            header += f" {a:>12}"
        print(header)
        print("  " + "-" * (25 + 13 * len(agents)))

        for metric_name, metric_key in [
            ("Avg tool calls", "tool_calls"),
            ("Avg total tokens", "total_tokens"),
            ("Avg cache-read tokens", "cache_read"),
        ]:
            line = f"  {metric_name:<25}"
            for a in agents:
                vals = [agent_data[a][iid][metric_key] for iid in shared]
                avg = statistics.mean(vals)
                if avg > 10000:
                    line += f" {avg:>11,.0f}"
                else:
                    line += f" {avg:>12.1f}"
            print(line)
        print()

        # Per-agent pairwise tool call deltas vs official
        print("  Tool call deltas vs official (per instance, averaged):")
        for agent in agents:
            if agent == "official":
                continue
            deltas = [agent_data[agent][iid]["tool_calls"] - agent_data["official"][iid]["tool_calls"]
                       for iid in shared]
            avg = statistics.mean(deltas)
            sign = "+" if avg > 0 else ""
            print(f"    {agent:>10}: {sign}{avg:.1f} tool calls per instance")
        print()

        print("  Token deltas vs official (per instance, averaged):")
        for agent in agents:
            if agent == "official":
                continue
            deltas = [agent_data[agent][iid]["total_tokens"] - agent_data["official"][iid]["total_tokens"]
                       for iid in shared]
            avg = statistics.mean(deltas)
            sign = "+" if avg > 0 else ""
            note = " (tokens not reported)" if avg == 0 and agent == "codex" else ""
            print(f"    {agent:>10}: {sign}{avg:,.0f} tokens per instance{note}")
        print()

    # --- 7. interesting patterns ---
    print("7. NOTABLE PATTERNS")
    print("-" * 60)

    # Check if any agents solved the exact same set
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            if resolved[a1] and resolved[a1] == resolved[a2]:
                print(f"  {a1} and {a2} solved the EXACT same instances")

    # Check overlap between agent pairs
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            both = resolved[a1] & resolved[a2]
            only_a1 = resolved[a1] - resolved[a2]
            only_a2 = resolved[a2] - resolved[a1]
            if only_a1 or only_a2:
                print(f"  {a1} vs {a2}: {len(both)} shared, {len(only_a1)} only-{a1}, {len(only_a2)} only-{a2}")

    # All-agent consensus
    all_resolved = set.intersection(*(resolved[a] for a in agents if resolved[a]))
    if all_resolved:
        print(f"\n  Solved by ALL agents ({len(all_resolved)}):")
        for iid in sorted(all_resolved):
            print(f"    {iid}")

    none_resolved = set()
    for iid in union:
        if all(iid not in resolved[a] for a in agents):
            none_resolved.add(iid)
    if none_resolved:
        print(f"\n  Solved by NO agent ({len(none_resolved)}):")
        for iid in sorted(none_resolved):
            print(f"    {iid}")

    print()
    print("=" * 80)
    print("Caveats:")
    print("  - Agents were run at different times; provider-side latency varies")
    print("  - Claude/Codex token counts use different accounting than OpenCode")
    print("  - Codex does not report token usage in its output")
    print("  - 3 instances had env errors across agents (astroid-1866, astroid-1978, pyvista-4315)")
    print("=" * 80)


if __name__ == "__main__":
    main()
