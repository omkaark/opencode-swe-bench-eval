#!/usr/bin/env python3
"""Run Claude Code or Codex on SWE-bench instances locally.

Usage:
    python run_cli_variant.py --cli claude --split dev --limit 23
    python run_cli_variant.py --cli codex --split dev --limit 23
    python run_cli_variant.py --cli claude --instance-id marshmallow-code__marshmallow-1343
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from datasets import load_dataset


def build_prompt(instance: dict, include_hints: bool = False) -> str:
    sections = [
        "You are working inside the root of a repository.",
        "Fix the issue described below by editing the repository in place.",
        "Keep changes focused, and run targeted checks only if they are practical.",
        "",
        "Issue:",
        instance["problem_statement"].strip(),
    ]
    fail_to_pass = []
    try:
        fail_to_pass = json.loads(instance.get("FAIL_TO_PASS", "[]"))
    except (json.JSONDecodeError, TypeError):
        pass
    if fail_to_pass:
        sections.extend(["", "Failing tests to target:"])
        sections.extend(f"- {t}" for t in fail_to_pass)
    sections.extend(["", "When you are done, stop without extra commentary."])
    return "\n".join(sections)


def capture_patch(repo_dir: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_dir, "--no-pager", "diff", "--binary"],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout


def run_instance(cli: str, instance: dict, output_dir: Path, timeout: int) -> dict:
    instance_id = instance["instance_id"]
    safe_name = instance_id.replace("/", "__")
    artifact_path = output_dir / f"{safe_name}.json"

    if artifact_path.exists():
        existing = json.loads(artifact_path.read_text())
        if not existing.get("error"):
            print(f"[skip] {cli} {instance_id} (exists)")
            return existing

    prompt = build_prompt(instance)

    # Clone repo to temp dir
    work_dir = tempfile.mkdtemp(prefix=f"swe-{safe_name}-")
    try:
        print(f"[start] {cli} {instance_id}")

        # Clone and checkout
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             f"https://github.com/{instance['repo']}.git", work_dir],
            capture_output=True, timeout=300, check=True,
        )
        subprocess.run(
            ["git", "-C", work_dir, "checkout", instance["base_commit"]],
            capture_output=True, timeout=120, check=True,
        )

        # Write prompt to file
        prompt_file = os.path.join(work_dir, ".swe_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)

        # Run the CLI agent
        start = time.time()
        if cli == "claude":
            cmd = [
                "claude", "-p", prompt,
                "--output-format", "json",
                "--max-turns", "50",
                "--allowedTools", "Edit,Read,Write,Bash,Glob,Grep",
            ]
        elif cli == "codex":
            cmd = [
                "codex", "exec",
                "--full-auto",
                "--skip-git-repo-check",
                "--json",
                prompt,
            ]
        else:
            raise ValueError(f"Unknown CLI: {cli}")

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=work_dir,
        )
        duration = time.time() - start

        diff = capture_patch(work_dir)
        git_status = subprocess.run(
            ["git", "-C", work_dir, "status", "--short"],
            capture_output=True, text=True, timeout=30,
        ).stdout

        payload = {
            "variant": cli,
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "cli_exit_code": result.returncode,
            "cli_output": result.stdout[:200000],  # cap at 200K chars
            "cli_stderr": result.stderr[:10000],
            "opencode_duration_seconds": round(duration, 2),
            "git_status": git_status,
            "model_patch": diff,
            "error": "",
        }

    except subprocess.TimeoutExpired:
        diff = capture_patch(work_dir) if os.path.exists(work_dir) else ""
        payload = {
            "variant": cli,
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "cli_exit_code": None,
            "cli_output": "",
            "cli_stderr": "",
            "opencode_duration_seconds": timeout,
            "git_status": "",
            "model_patch": diff,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as e:
        payload = {
            "variant": cli,
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "cli_exit_code": None,
            "cli_output": "",
            "cli_stderr": "",
            "opencode_duration_seconds": 0,
            "git_status": "",
            "model_patch": "",
            "error": str(e),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Save artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n")

    patch_chars = len(payload["model_patch"])
    dur = payload["opencode_duration_seconds"]
    print(f"[done]  {cli} {instance_id} patch={patch_chars} chars  dur={dur}s")
    return payload


def write_predictions(output_dir: Path, cli: str, results: list[dict]) -> Path:
    path = output_dir / f"predictions_{cli}.jsonl"
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "model_name_or_path": f"opencode-{cli}",
                "model_patch": r["model_patch"],
            }) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, choices=["claude", "codex"])
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    items = [dict(item) for item in dataset]

    if args.instance_ids:
        wanted = set(args.instance_ids)
        items = [i for i in items if i["instance_id"] in wanted]
    if args.limit:
        items = items[:args.limit]

    output_dir = Path(args.output_dir) / args.cli
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.cli} on {len(items)} instances")

    results = []
    for instance in items:
        result = run_instance(args.cli, instance, output_dir, args.timeout)
        results.append(result)

    pred_path = write_predictions(Path(args.output_dir), args.cli, results)
    print(f"\nPredictions: {pred_path}")
    print(f"Resolved patches: {sum(1 for r in results if r['model_patch'])}/{len(results)}")


if __name__ == "__main__":
    main()
