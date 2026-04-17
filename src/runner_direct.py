"""Direct-sandbox runner — bypasses swerex tunnels entirely.

Runs opencode inside Modal sandboxes using script(1) for PTY,
streams stdout directly via Modal's gRPC API. No HTTP tunnel.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import modal
import yaml
from datasets import load_dataset

from .images import load_variants
from .utils import save_artifact, save_log, save_predictions

DATASET = "princeton-nlp/SWE-bench_Lite"

modal.enable_output()


def build_prompt(instance: dict[str, Any]) -> str:
    sections = [
        f"Please fix the following issue in the {instance['repo']} repository.",
        f"\n## Problem Statement\n{instance['problem_statement']}",
    ]
    if instance.get("hints_text"):
        sections.append(f"\n## Hints\n{instance['hints_text']}")
    return "\n".join(sections)


def make_run_script(instance: dict[str, Any], model: str) -> str:
    repo = instance["repo"]
    commit = instance["base_commit"]
    return f"""
exec 2>&1
git clone --filter=blob:none --no-checkout https://github.com/{repo}.git /repo || exit 1
cd /repo
git checkout {commit} || exit 1
cat > /tmp/problem.txt << 'PROBLEM_EOF'
{build_prompt(instance)}
PROBLEM_EOF
cat > /tmp/run_opencode.sh << 'SCRIPT_EOF'
cd /repo && opencode run --agent build --model {model} --format json "$(cat /tmp/problem.txt)"
SCRIPT_EOF
chmod +x /tmp/run_opencode.sh
echo "=== OPENCODE_OUTPUT_START ==="
script -qec "bash /tmp/run_opencode.sh" /dev/null
OPENCODE_EXIT=$?
echo ""
echo "=== OPENCODE_OUTPUT_END ==="
echo "=== PATCH_START ==="
cd /repo && git diff 2>/dev/null
echo "=== PATCH_END ==="
echo "=== EXIT_CODE=$OPENCODE_EXIT ==="
"""


async def run_instance(
    variant: str,
    image: modal.Image,
    instance: dict[str, Any],
    secret: modal.Secret,
    model: str,
    output_dir: Path,
    app: modal.App,
    timeout: int = 300,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    script = make_run_script(instance, model)

    t0 = time.time()
    try:
        sb = await modal.Sandbox.create.aio(
            "bash", "-c", script,
            image=image,
            timeout=timeout,
            secrets=[secret],
            app=app,
        )

        ndjson_lines = []
        patch_lines = []
        in_opencode = False
        in_patch = False

        async for line in sb.stdout:
            if "OPENCODE_OUTPUT_START" in line:
                in_opencode = True
                continue
            if "OPENCODE_OUTPUT_END" in line:
                in_opencode = False
                continue
            if "PATCH_START" in line:
                in_patch = True
                continue
            if "PATCH_END" in line:
                in_patch = False
                continue
            if in_opencode:
                ndjson_lines.append(line.rstrip())
            elif in_patch:
                patch_lines.append(line.rstrip())

        try:
            await sb.wait.aio()
        except modal.exception.SandboxTimeoutError:
            print(f"[timeout] {variant} {instance_id}")

    except Exception as e:
        print(f"[error]  {variant} {instance_id}: {e}")
        ndjson_lines = []
        patch_lines = []

    duration = round(time.time() - t0, 2)
    opencode_output = "\n".join(ndjson_lines)
    patch = "\n".join(patch_lines).strip()

    # Parse token stats from NDJSON
    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for line in ndjson_lines:
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
            if e.get("type") == "step_finish":
                t = e.get("part", {}).get("tokens", {})
                tokens["input"] += t.get("input", 0)
                tokens["output"] += t.get("output", 0)
                c = t.get("cache", {})
                tokens["cache_read"] += c.get("read", 0)
                tokens["cache_write"] += c.get("write", 0)
        except json.JSONDecodeError:
            continue

    save_log(output_dir, variant, instance_id, opencode_output)

    result = {
        "instance_id": instance_id,
        "variant": variant,
        "model_patch": patch,
        "duration": duration,
        "input": tokens["input"],
        "output": tokens["output"],
        "cache_read": tokens["cache_read"],
        "cache_write": tokens["cache_write"],
        "cost": 0,
    }
    safe_name = instance_id.replace("/", "__")
    artifact_path = output_dir / variant / f"{safe_name}.json"
    save_artifact(artifact_path, result)
    print(f"[done]  {variant} {instance_id} patch_chars={len(patch)}")
    return result


async def run_variant(
    variant: str,
    image: modal.Image,
    instances: list[dict[str, Any]],
    secret: modal.Secret,
    model: str,
    concurrency: int,
    output_dir: Path,
    app: modal.App,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def run_with_limit(inst):
        async with sem:
            return await run_instance(variant, image, inst, secret, model, output_dir, app)

    return await asyncio.gather(*(run_with_limit(i) for i in instances))


async def main(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    split = config.get("split", "dev")
    output_dir = Path(config.get("output_dir", "outputs")).resolve()
    concurrency = config.get("concurrency", 8)
    model = config.get("model")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    limit = config.get("limit")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"Error: {api_key_env} not set")
        return

    instances = [dict(item) for item in load_dataset(DATASET, split=split)]
    if limit:
        instances = instances[:limit]

    secret = modal.Secret.from_dict({api_key_env: api_key})
    variant_images = load_variants(config)
    app = await modal.App.lookup.aio("swe-rex", create_if_missing=True)

    for variant, image in variant_images.items():
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant} ({len(instances)} instances, concurrency={concurrency})")
        print(f"{'='*60}")
        results = await run_variant(
            variant, image, instances, secret, model, concurrency, output_dir, app,
        )
        predictions_path = save_predictions(output_dir, variant, results)
        print(f"[predictions] {variant}: {predictions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.config))
