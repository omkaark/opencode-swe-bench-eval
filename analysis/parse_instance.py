#!/usr/bin/env python3
"""Parse a single instance JSON artifact into a structured summary."""

import json
import sys
from pathlib import Path


FAILURE_SIGNALS = [
    "No files found",
    "No matches found",
    "No match found",
    "not found",
    "command not found",
]


def is_failed_tool_call(output: str) -> bool:
    if not output or not output.strip():
        return True
    lower = output.lower()
    return any(sig.lower() in lower for sig in FAILURE_SIGNALS)


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


def parse_instance(artifact: dict) -> dict:
    events = parse_ndjson(artifact.get("opencode_output", ""))

    # Counters
    step_count = 0
    tool_counts: dict[str, int] = {}
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    reasoning_tokens = 0

    files_read: list[str] = []
    files_edited: list[str] = []
    failed_calls: list[dict] = []
    all_tool_calls: list[dict] = []

    first_timestamp = None
    first_edit_timestamp = None
    step_index = 0

    for event in events:
        etype = event.get("type")
        part = event.get("part", {})
        ts = event.get("timestamp")

        if first_timestamp is None and ts is not None:
            first_timestamp = ts

        if etype == "step_start":
            step_count += 1
            step_index = step_count

        elif etype == "tool_use":
            tool = part.get("tool", "unknown")
            state = part.get("state", {})
            inp = state.get("input", {})
            output = state.get("output", "") or ""
            time_info = state.get("time", {})

            tool_counts[tool] = tool_counts.get(tool, 0) + 1

            call_record = {
                "step": step_index,
                "tool": tool,
                "input": inp,
                "output_preview": output[:200],
                "failed": is_failed_tool_call(output),
                "timestamp": ts,
                "duration_ms": (
                    time_info.get("end", 0) - time_info.get("start", 0)
                    if time_info.get("start") and time_info.get("end")
                    else None
                ),
            }
            all_tool_calls.append(call_record)

            if tool == "read":
                fp = inp.get("filePath", "")
                if fp and fp not in files_read:
                    files_read.append(fp)

            elif tool == "edit":
                fp = inp.get("filePath", "")
                if fp and fp not in files_edited:
                    files_edited.append(fp)
                if first_edit_timestamp is None:
                    first_edit_timestamp = ts

            elif tool == "glob":
                pass  # handled in tree builder

            if call_record["failed"]:
                failed_calls.append(call_record)

        elif etype == "step_finish":
            tokens = part.get("tokens", {})
            total_tokens += tokens.get("total", 0)
            input_tokens += tokens.get("input", 0)
            output_tokens += tokens.get("output", 0)
            reasoning_tokens += tokens.get("reasoning", 0)
            cache = tokens.get("cache", {})
            cache_read += cache.get("read", 0)
            cache_write += cache.get("write", 0)

    # Time to first edit
    time_to_first_edit = None
    if first_timestamp is not None and first_edit_timestamp is not None:
        time_to_first_edit = (first_edit_timestamp - first_timestamp) / 1000.0

    # Dead ends: consecutive failed calls before a success
    dead_ends = []
    run: list[dict] = []
    for call in all_tool_calls:
        if call["failed"]:
            run.append(call)
        else:
            if len(run) >= 2:
                dead_ends.append({
                    "length": len(run),
                    "steps": [c["step"] for c in run],
                    "tools": [c["tool"] for c in run],
                    "inputs_preview": [
                        json.dumps(c["input"])[:100] for c in run
                    ],
                })
            run = []
    if len(run) >= 2:
        dead_ends.append({
            "length": len(run),
            "steps": [c["step"] for c in run],
            "tools": [c["tool"] for c in run],
            "inputs_preview": [
                json.dumps(c["input"])[:100] for c in run
            ],
        })

    summary = {
        "instance_id": artifact.get("instance_id", ""),
        "variant": artifact.get("variant", ""),
        "repo": artifact.get("repo", ""),
        "tokens": {
            "total": total_tokens,
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "cache_read": cache_read,
            "cache_write": cache_write,
        },
        "duration_seconds": artifact.get("opencode_duration_seconds"),
        "time_to_first_edit_seconds": time_to_first_edit,
        "steps": step_count,
        "tool_calls": {
            "total": sum(tool_counts.values()),
            "by_tool": dict(sorted(tool_counts.items())),
        },
        "files_read": files_read,
        "files_edited": files_edited,
        "failed_tool_calls": len(failed_calls),
        "failed_calls_detail": [
            {
                "step": c["step"],
                "tool": c["tool"],
                "input": c["input"],
                "output_preview": c["output_preview"],
            }
            for c in failed_calls
        ],
        "dead_ends": dead_ends,
        "patch_non_empty": bool(artifact.get("model_patch", "").strip()),
        "error": artifact.get("error", ""),
    }
    return summary


def print_summary(s: dict) -> None:
    print(f"Instance: {s['instance_id']} ({s['variant']})")
    print(f"Repo:     {s['repo']}")
    print()

    t = s["tokens"]
    print(f"Tokens:   {t['total']:,} total")
    print(f"          {t['input']:,} input / {t['output']:,} output / {t['reasoning']:,} reasoning")
    print(f"          {t['cache_read']:,} cache read / {t['cache_write']:,} cache write")
    print()

    dur = s["duration_seconds"]
    print(f"Duration: {dur}s" if dur else "Duration: N/A")
    ttfe = s["time_to_first_edit_seconds"]
    print(f"Time to first edit: {ttfe:.1f}s" if ttfe else "Time to first edit: N/A (no edits)")
    print()

    tc = s["tool_calls"]
    print(f"Steps: {s['steps']}  |  Tool calls: {tc['total']}")
    for tool, count in tc["by_tool"].items():
        print(f"  {tool}: {count}")
    print()

    print(f"Files read ({len(s['files_read'])}):")
    for f in s["files_read"]:
        print(f"  {f}")
    print(f"Files edited ({len(s['files_edited'])}):")
    for f in s["files_edited"]:
        print(f"  {f}")
    print()

    print(f"Failed tool calls: {s['failed_tool_calls']}")
    print(f"Dead ends (consecutive failures >= 2): {len(s['dead_ends'])}")
    for de in s["dead_ends"]:
        print(f"  {de['length']} calls in steps {de['steps']} using {de['tools']}")
    print()

    print(f"Patch non-empty: {s['patch_non_empty']}")
    if s["error"]:
        print(f"Error: {s['error'][:200]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_instance.py <instance.json> [--quiet]", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    quiet = "--quiet" in sys.argv

    with open(path) as f:
        artifact = json.load(f)

    summary = parse_instance(artifact)

    # Save parsed JSON
    out_path = path.with_name(path.stem + "_parsed.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    if not quiet:
        print_summary(summary)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
