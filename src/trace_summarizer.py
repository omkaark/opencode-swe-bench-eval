"""Parse instance artifacts into proposer-friendly summaries. Multi-session aware."""
import json
from collections import Counter


def parse_ndjson(raw: str) -> list[dict]:
    events = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _empty_session_stats() -> dict:
    return {
        "steps": 0,
        "tool_calls": [],
        "tokens": {
            "total": 0, "input": 0, "output": 0, "reasoning": 0,
            "cache_read": 0, "cache_write": 0,
        },
        "files_read": [],
        "files_edited": [],
        "read_counts": Counter(),
        "grep_patterns": Counter(),
        "first_step_tokens": None,
    }


def summarize_instance(artifact: dict) -> dict:
    """Per-session breakdown under `sessions` (main + subagent_N). Flat totals kept for backward compat."""
    events = parse_ndjson(artifact.get("opencode_output", ""))

    by_session: dict[str, dict] = {}
    session_order: list[str] = []

    def _ensure(sid: str) -> dict:
        if sid not in by_session:
            by_session[sid] = _empty_session_stats()
            session_order.append(sid)
        return by_session[sid]

    for event in events:
        etype = event.get("type")
        part = event.get("part", {})
        sid = event.get("sessionID") or (part.get("sessionID") if isinstance(part, dict) else None) or "__unknown__"
        s = _ensure(sid)

        if etype == "step_start":
            s["steps"] += 1

        elif etype == "tool_use":
            tool = part.get("tool", "unknown")
            state = part.get("state", {})
            inp = state.get("input", {})
            output = state.get("output", "") or ""

            s["tool_calls"].append({"tool": tool, "input": inp, "output_preview": output[:200]})

            if tool == "read":
                fp = inp.get("filePath", "")
                if fp:
                    s["read_counts"][fp] += 1
                    if fp not in s["files_read"]:
                        s["files_read"].append(fp)
            elif tool == "edit":
                fp = inp.get("filePath", "")
                if fp and fp not in s["files_edited"]:
                    s["files_edited"].append(fp)
            elif tool == "grep":
                pattern = inp.get("pattern", "")
                if pattern:
                    s["grep_patterns"][pattern] += 1

        elif etype == "step_finish":
            tokens = part.get("tokens", {})
            cache = tokens.get("cache", {})
            step_tokens = {
                "total": tokens.get("total", 0),
                "input": tokens.get("input", 0),
                "output": tokens.get("output", 0),
                "reasoning": tokens.get("reasoning", 0),
                "cache_read": cache.get("read", 0),
                "cache_write": cache.get("write", 0),
            }
            if s["first_step_tokens"] is None:
                s["first_step_tokens"] = step_tokens
            for k, v in step_tokens.items():
                s["tokens"][k] += v

    labeled: dict[str, dict] = {}
    for i, sid in enumerate(session_order):
        label = "main" if i == 0 else f"subagent_{i}"
        s = by_session[sid]
        redundant = []
        for fp, count in s["read_counts"].items():
            if count > 1:
                redundant.append({"tool": "read", "file": fp, "count": count})
        for pat, count in s["grep_patterns"].items():
            if count > 1:
                redundant.append({"tool": "grep", "pattern": pat, "count": count})
        labeled[label] = {
            "session_id": sid,
            "steps": s["steps"],
            "total_tool_calls": len(s["tool_calls"]),
            "tokens": s["tokens"],
            "first_step_tokens": s["first_step_tokens"],
            "files_read": s["files_read"],
            "files_edited": s["files_edited"],
            "redundant_calls": redundant,
            "tool_sequence": [{"tool": c["tool"], "input": c["input"]} for c in s["tool_calls"]],
        }

    flat_tokens = {k: 0 for k in ["total", "input", "output", "reasoning", "cache_read", "cache_write"]}
    flat_files_read: list[str] = []
    flat_files_edited: list[str] = []
    flat_redundant: list[dict] = []
    total_tool_calls = 0
    total_steps = 0
    for sess in labeled.values():
        total_tool_calls += sess["total_tool_calls"]
        total_steps += sess["steps"]
        for k in flat_tokens:
            flat_tokens[k] += sess["tokens"][k]
        for fp in sess["files_read"]:
            if fp not in flat_files_read:
                flat_files_read.append(fp)
        for fp in sess["files_edited"]:
            if fp not in flat_files_edited:
                flat_files_edited.append(fp)
        flat_redundant.extend(sess["redundant_calls"])

    main = labeled.get("main", {})
    return {
        "instance_id": artifact.get("instance_id", ""),
        "duration_seconds": artifact.get("opencode_duration_seconds"),
        "patch_non_empty": bool(artifact.get("model_patch", "").strip()),
        "error": artifact.get("error", ""),

        "steps": total_steps,
        "total_tool_calls": total_tool_calls,
        "tokens": flat_tokens,
        "files_read": flat_files_read,
        "files_edited": flat_files_edited,
        "redundant_calls": flat_redundant,
        "tool_sequence": main.get("tool_sequence", []),
        "session_count": len(labeled),
        "sessions": labeled,
    }


def analyze_kv_cache(artifact: dict) -> list[dict]:
    """Per-session cache verdict from first step: hit (>=0.8), miss (<=0.2), partial, or no_data."""
    summary = summarize_instance(artifact)
    out = []
    for label, sess in summary["sessions"].items():
        fst = sess.get("first_step_tokens") or {}
        inp = fst.get("input", 0)
        cache_read = fst.get("cache_read", 0)
        ratio = (cache_read / inp) if inp > 0 else 0.0
        if inp == 0:
            verdict = "no_data"
        elif ratio >= 0.8:
            verdict = "cache_hit"
        elif ratio <= 0.2:
            verdict = "cache_miss"
        else:
            verdict = "partial"
        out.append({
            "label": label,
            "session_id": sess["session_id"],
            "first_input_tokens": inp,
            "first_cache_read": cache_read,
            "cache_ratio": round(ratio, 3),
            "verdict": verdict,
        })
    return out


def summarize_run(artifact_dir: str, resolved_ids: set[str]) -> dict:
    """Summarize an entire run (all instances) for the proposer."""
    from pathlib import Path
    summaries = []
    for p in sorted(Path(artifact_dir).glob("*.json")):
        if p.stem.endswith("_parsed") or p.stem.endswith("_tree"):
            continue
        artifact = json.loads(p.read_text())
        s = summarize_instance(artifact)
        s["resolved"] = s["instance_id"] in resolved_ids
        summaries.append(s)

    resolved_list = [s for s in summaries if s["resolved"]]
    unresolved_list = [s for s in summaries if not s["resolved"] and not s["error"]]
    errored_list = [s for s in summaries if s["error"]]

    non_errored = [s for s in summaries if not s["error"]]
    avg_duration = round(sum(s["duration_seconds"] or 0 for s in non_errored) / max(len(non_errored), 1), 1)
    avg_tool_calls = round(sum(s["total_tool_calls"] for s in non_errored) / max(len(non_errored), 1), 1)
    avg_tokens = round(sum(s["tokens"]["total"] for s in non_errored) / max(len(non_errored), 1))

    return {
        "total_instances": len(summaries),
        "resolved": len(resolved_list),
        "unresolved": len(unresolved_list),
        "errored": len(errored_list),
        "resolve_rate": len(resolved_list) / len(summaries) if summaries else 0,
        "avg_duration": avg_duration,
        "avg_tool_calls": avg_tool_calls,
        "avg_tokens": avg_tokens,
        "per_instance": summaries,
    }


def format_for_proposer(run_summary: dict, full_trace_ids: list[str] | None = None, artifacts_dir: str | None = None) -> str:
    """Format a run summary as text for the proposer's context."""
    lines = []
    lines.append(f"## Run Results: {run_summary['resolved']}/{run_summary['total_instances']} resolved ({run_summary['resolve_rate']:.1%})")
    lines.append(f"Avg duration: {run_summary['avg_duration']}s | Avg tool calls: {run_summary['avg_tool_calls']}")
    lines.append("")

    lines.append("### Resolved instances:")
    for s in run_summary["per_instance"]:
        if s["resolved"]:
            lines.append(f"  - {s['instance_id']}: {s['duration_seconds']}s, {s['total_tool_calls']} tools, {len(s['redundant_calls'])} redundant")
    lines.append("")

    lines.append("### Unresolved instances:")
    for s in run_summary["per_instance"]:
        if not s["resolved"] and not s["error"]:
            redundant_str = ", ".join(f"{r['tool']}({r.get('file', r.get('pattern', '?'))})x{r['count']}" for r in s["redundant_calls"])
            lines.append(f"  - {s['instance_id']}: {s['duration_seconds']}s, {s['total_tool_calls']} tools, redundant=[{redundant_str}]")
    lines.append("")

    if run_summary.get("errored"):
        lines.append("### Errored instances:")
        for s in run_summary["per_instance"]:
            if s["error"]:
                lines.append(f"  - {s['instance_id']}: {s['error'][:100]}")
        lines.append("")

    # Append full traces for selected instances
    if full_trace_ids and artifacts_dir:
        from pathlib import Path
        lines.append("### Full traces for key instances:")
        for iid in full_trace_ids:
            safe = iid.replace("/", "__")
            p = Path(artifacts_dir) / f"{safe}.json"
            if p.exists():
                artifact = json.loads(p.read_text())
                lines.append(f"\n#### {iid}")
                lines.append("```ndjson")
                lines.append(artifact.get("opencode_output", "")[:15000])
                lines.append("```")
        lines.append("")

    return "\n".join(lines)
