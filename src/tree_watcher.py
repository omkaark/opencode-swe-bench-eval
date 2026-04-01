#!/usr/bin/env python3
"""Runs inside the Modal sandbox. Tails opencode NDJSON output and
continuously writes a compact tree context to /tmp/tree_context.json.

Usage:
    python3 tree_watcher.py /tmp/opencode_output.ndjson &

The watcher follows the file (like tail -f), parses each NDJSON line,
builds the tree incrementally, and writes the compact context after
every tool_use or step_finish event so the tree is always up to date
when a subagent reads it.
"""

import json
import os
import sys
import time

TREE_CONTEXT_PATH = "/tmp/tree_context.json"
FAILURE_SIGNALS = ["no files found", "no matches found", "no match found",
                   "not found", "command not found"]


def is_failed(output):
    if not output or not output.strip():
        return True
    lower = output.lower()
    return any(sig in lower for sig in FAILURE_SIGNALS)


class LiveTreeBuilder:
    def __init__(self):
        self.repo_map = []
        self.files_examined = {}
        self.failed_searches = []
        self.successful_searches = []
        self.step_index = 0

    def feed_line(self, line):
        line = line.strip()
        if not line or not line.startswith("{"):
            return False
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        return self.feed_event(event)

    def feed_event(self, event):
        etype = event.get("type")
        part = event.get("part", {})

        if etype == "step_start":
            self.step_index += 1
            return False

        if etype == "tool_use":
            self._handle_tool(part)
            return True  # tree updated, write it out

        if etype == "step_finish":
            return True  # good checkpoint to write

        return False

    @staticmethod
    def _extract_signatures(output):
        if not output:
            return []
        sigs = []
        for line in output.split("\n"):
            stripped = line.strip()
            if ":" in stripped and stripped.split(":")[0].strip().isdigit():
                stripped = stripped.split(":", 1)[1].strip()
            if stripped.startswith(("def ", "class ", "async def ")):
                sigs.append(stripped.rstrip(":").strip())
        return sigs[:20]

    def _handle_tool(self, part):
        tool = part.get("tool", "")
        state = part.get("state", {})
        inp = state.get("input", {})
        output = state.get("output", "") or ""

        if tool == "glob":
            for f in output.strip().split("\n"):
                f = f.strip()
                if f and f.startswith("/") and f not in self.repo_map:
                    self.repo_map.append(f)

        elif tool == "read":
            fp = inp.get("filePath", "")
            if fp:
                entry = self.files_examined.setdefault(fp, {"read_count": 0})
                entry["read_count"] += 1
                # Extract function/class signatures from read content
                sigs = self._extract_signatures(output)
                if sigs:
                    existing = entry.setdefault("signatures", [])
                    seen = set(existing)
                    for s in sigs:
                        if s not in seen:
                            existing.append(s)
                            seen.add(s)

        elif tool == "edit":
            fp = inp.get("filePath", "")
            if fp:
                entry = self.files_examined.setdefault(fp, {"read_count": 0})
                edits = entry.setdefault("edits", [])
                edits.append({
                    "old": inp.get("oldString", "")[:500],
                    "new": inp.get("newString", "")[:500],
                })

        if is_failed(output):
            self.failed_searches.append({
                "tool": tool,
                "input": inp,
            })
        elif tool in ("grep", "bash"):
            self.successful_searches.append({
                "tool": tool,
                "input": inp,
                "output": output[:200],
            })

    def to_compact(self):
        return json.dumps({
            "repo_map": self.repo_map,
            "files_examined": self.files_examined,
            "failed_searches": self.failed_searches,
            "successful_searches": self.successful_searches,
        }, separators=(",", ":"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tree_watcher.py <ndjson_file>", file=sys.stderr)
        sys.exit(1)

    ndjson_path = sys.argv[1]
    builder = LiveTreeBuilder()

    # Wait for file to exist
    while not os.path.exists(ndjson_path):
        time.sleep(0.1)

    with open(ndjson_path, "r") as f:
        while True:
            line = f.readline()
            if line:
                updated = builder.feed_line(line)
                if updated:
                    # Atomic write: write to tmp then rename
                    tmp = TREE_CONTEXT_PATH + ".tmp"
                    with open(tmp, "w") as out:
                        out.write(builder.to_compact())
                    os.rename(tmp, TREE_CONTEXT_PATH)
            else:
                # No new data — check if opencode is still running
                # by looking for a sentinel file
                if os.path.exists("/tmp/opencode_done"):
                    break
                time.sleep(0.05)


if __name__ == "__main__":
    main()
