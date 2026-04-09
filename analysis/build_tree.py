#!/usr/bin/env python3
"""Build a structured tree from a single instance JSON artifact."""

import json
import sys
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tree_builder import TreeBuilder


def build_tree_from_artifact(artifact: dict) -> dict:
    builder = TreeBuilder(
        instance_id=artifact.get("instance_id", ""),
        variant=artifact.get("variant", ""),
        repo=artifact.get("repo", ""),
    )
    for line in (artifact.get("opencode_output", "") or "").split("\n"):
        builder.feed_line(line)

    tree = builder.to_dict()
    tree["total_duration_seconds"] = artifact.get("opencode_duration_seconds")
    tree["patch_non_empty"] = bool(artifact.get("model_patch", "").strip())
    return tree


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_tree.py <instance.json>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    with open(path) as f:
        artifact = json.load(f)

    tree = build_tree_from_artifact(artifact)

    out_path = path.with_name(path.stem + "_tree.json")
    out_path.write_text(json.dumps(tree, indent=2) + "\n")

    # Print summary
    print(f"Instance: {tree['instance_id']} ({tree['variant']})")
    print(f"Files discovered (glob): {len(tree['repo_map'])}")
    print(f"Files with reads/edits:  {len(tree['per_file'])}")
    for fp, info in tree["per_file"].items():
        edits_str = f", {len(info['edits'])} edits" if info["edits"] else ""
        print(f"  {fp}: {len(info['reads'])} reads{edits_str}")
    print(f"Failed approaches:  {len(tree['failed_approaches'])}")
    print(f"Successful approaches: {len(tree['successful_approaches'])}")
    t = tree["token_cost"]
    print(f"Tokens: {t['total']:,} total ({t['cache_read']:,} cached)")
    print(f"Duration: {tree['total_duration_seconds']}s")
    ttfe = tree["time_to_first_edit_seconds"]
    print(f"Time to first edit: {ttfe}s" if ttfe else "Time to first edit: N/A")

    # Show compact context size
    builder = TreeBuilder(
        instance_id=tree["instance_id"],
        variant=tree["variant"],
        repo=tree["repo"],
    )
    for line in (artifact.get("opencode_output", "") or "").split("\n"):
        builder.feed_line(line)
    compact = builder.to_compact_context()
    print(f"Compact context size: {len(compact):,} chars (~{len(compact)//4:,} tokens)")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
