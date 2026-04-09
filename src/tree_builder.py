"""Incremental tree builder that processes NDJSON events one at a time.

Used by:
- build_tree.py (offline, from saved artifacts)
- runner.py (live, line-by-line as opencode runs)
"""

import json


FAILURE_SIGNALS = [
    "No files found",
    "No matches found",
    "No match found",
    "not found",
    "command not found",
]


def is_failed(output: str) -> bool:
    if not output or not output.strip():
        return True
    lower = output.lower()
    return any(sig.lower() in lower for sig in FAILURE_SIGNALS)


class TreeBuilder:
    """Incrementally builds a structured tree from opencode NDJSON events."""

    def __init__(self, instance_id: str = "", variant: str = "", repo: str = ""):
        self.instance_id = instance_id
        self.variant = variant
        self.repo = repo

        self.repo_map: list[str] = []
        self.per_file: dict[str, dict] = {}
        self.failed_approaches: list[dict] = []
        self.successful_approaches: list[dict] = []

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0

        self.first_timestamp: int | None = None
        self.first_edit_timestamp: int | None = None
        self.step_index = 0

    def _ensure_file(self, fp: str) -> dict:
        if fp not in self.per_file:
            self.per_file[fp] = {
                "reads": [],
                "edits": [],
                "first_read_step": None,
                "first_edit_step": None,
            }
        return self.per_file[fp]

    def feed_line(self, line: str) -> None:
        """Process a single NDJSON line. Non-JSON lines are silently ignored."""
        line = line.strip()
        if not line or not line.startswith("{"):
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        self.feed_event(event)

    def feed_event(self, event: dict) -> None:
        """Process a single parsed event dict."""
        etype = event.get("type")
        part = event.get("part", {})
        ts = event.get("timestamp")

        if self.first_timestamp is None and ts is not None:
            self.first_timestamp = ts

        if etype == "step_start":
            self.step_index += 1

        elif etype == "tool_use":
            self._handle_tool_use(part, ts)

        elif etype == "step_finish":
            tokens = part.get("tokens", {})
            self.total_tokens += tokens.get("total", 0)
            self.input_tokens += tokens.get("input", 0)
            self.output_tokens += tokens.get("output", 0)
            cache = tokens.get("cache", {})
            self.cache_read += cache.get("read", 0)
            self.cache_write += cache.get("write", 0)

    @staticmethod
    def _extract_signatures(output: str, filepath: str, inp: dict) -> dict | None:
        """Extract function/class signatures and key lines from a read output."""
        if not output:
            return None
        lines = output.split("\n")
        signatures = []
        offset = inp.get("offset", 0) or 0
        for line in lines:
            stripped = line.strip()
            # Strip line numbers (format: "123:    def foo(...")
            if ":" in stripped and stripped.split(":")[0].strip().isdigit():
                stripped = stripped.split(":", 1)[1].strip()
            # Capture function and class definitions
            if stripped.startswith(("def ", "class ", "async def ")):
                signatures.append(stripped.rstrip(":").strip())
        if not signatures:
            return None
        return {
            "offset": offset,
            "limit": inp.get("limit"),
            "signatures": signatures[:20],  # cap at 20 per read
        }

    def _handle_tool_use(self, part: dict, ts: int | None) -> None:
        tool = part.get("tool", "unknown")
        state = part.get("state", {})
        inp = state.get("input", {})
        output = state.get("output", "") or ""

        if tool == "glob":
            for file_line in output.strip().split("\n"):
                file_line = file_line.strip()
                if file_line and file_line.startswith("/") and file_line not in self.repo_map:
                    self.repo_map.append(file_line)

        elif tool == "read":
            fp = inp.get("filePath", "")
            if fp:
                entry = self._ensure_file(fp)
                entry["reads"].append(self.step_index)
                if entry["first_read_step"] is None:
                    entry["first_read_step"] = self.step_index
                # Capture key content: function/class signatures from the read
                if "content_snippets" not in entry:
                    entry["content_snippets"] = []
                snippet = self._extract_signatures(output, fp, inp)
                if snippet:
                    entry["content_snippets"].append(snippet)

        elif tool == "edit":
            fp = inp.get("filePath", "")
            if fp:
                entry = self._ensure_file(fp)
                entry["edits"].append({
                    "step": self.step_index,
                    "old": inp.get("oldString", ""),
                    "new": inp.get("newString", ""),
                })
                if entry["first_edit_step"] is None:
                    entry["first_edit_step"] = self.step_index
                if self.first_edit_timestamp is None:
                    self.first_edit_timestamp = ts

        if is_failed(output):
            reason = "empty output"
            lower = output.lower()
            for sig in FAILURE_SIGNALS:
                if sig.lower() in lower:
                    reason = sig.lower()
                    break
            self.failed_approaches.append({
                "step": self.step_index,
                "tool": tool,
                "input": inp,
                "output_preview": output[:300],
                "reason_failed": reason,
            })
        elif tool in ("grep", "bash", "edit"):
            self.successful_approaches.append({
                "step": self.step_index,
                "tool": tool,
                "input": inp,
                "output_preview": output[:300],
            })

    def to_dict(self) -> dict:
        """Return the full tree as a dict."""
        time_to_first_edit = None
        if self.first_timestamp is not None and self.first_edit_timestamp is not None:
            time_to_first_edit = round(
                (self.first_edit_timestamp - self.first_timestamp) / 1000.0, 2
            )

        return {
            "instance_id": self.instance_id,
            "variant": self.variant,
            "repo": self.repo,
            "repo_map": self.repo_map,
            "per_file": self.per_file,
            "failed_approaches": self.failed_approaches,
            "successful_approaches": self.successful_approaches,
            "token_cost": {
                "total": self.total_tokens,
                "input": self.input_tokens,
                "output": self.output_tokens,
                "cache_read": self.cache_read,
                "cache_write": self.cache_write,
            },
            "time_to_first_edit_seconds": time_to_first_edit,
        }

    def to_compact_context(self) -> str:
        """Serialize the tree as a compact string for passing to a subagent.

        This is the key output: instead of forwarding the full conversation
        prefix (~22k tokens), the subagent gets this compact summary (~2k tokens).
        """
        tree = self.to_dict()

        # Strip verbose fields that don't help the subagent
        compact = {
            "repo_map": tree["repo_map"],
            "files_examined": {},
            "failed_searches": [],
            "successful_searches": [],
        }

        for fp, info in tree["per_file"].items():
            entry: dict = {"read_count": len(info["reads"])}
            # Include function/class signatures found in this file
            all_sigs: list[str] = []
            for snippet in info.get("content_snippets", []):
                if snippet and "signatures" in snippet:
                    all_sigs.extend(snippet["signatures"])
            if all_sigs:
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique_sigs = []
                for s in all_sigs:
                    if s not in seen:
                        seen.add(s)
                        unique_sigs.append(s)
                entry["signatures"] = unique_sigs[:30]
            if info["edits"]:
                entry["edits"] = [
                    {"old": e["old"][:500], "new": e["new"][:500]}
                    for e in info["edits"]
                ]
            compact["files_examined"][fp] = entry

        # Compact failed searches: just tool + input, no output
        for fa in tree["failed_approaches"]:
            compact["failed_searches"].append({
                "tool": fa["tool"],
                "input": fa["input"],
            })

        # Compact successful searches: tool + input + truncated output
        for sa in tree["successful_approaches"]:
            compact["successful_searches"].append({
                "tool": sa["tool"],
                "input": sa["input"],
                "output": sa["output_preview"][:200],
            })

        return json.dumps(compact, separators=(",", ":"))
