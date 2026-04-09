import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import modal
from datasets import load_dataset
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import BashAction, Command, CreateBashSessionRequest, ReadFileRequest, WriteFileRequest

from images import CONFIG, VARIANT_IMAGES
from utils import save_artifact, save_log, save_predictions

DATASET = "princeton-nlp/SWE-bench_Lite"
CLONE_TIMEOUT = 300
COMMAND_TIMEOUT = 1200
STARTUP_TIMEOUT = 180
RUNTIME_TIMEOUT = 1800
DEPLOYMENT_TIMEOUT = 3600
API_KEY_ENV_VAR = "OPENCODE_API_KEY"

def build_prompt(instance: dict[str, Any]) -> str:
    sections = [
        "You are working inside the root of a repository.",
        "Fix the issue described below by editing the repository in place.",
        "Keep changes focused, and run targeted checks only if they are practical in this environment.",
        "",
        "Issue:",
        instance["problem_statement"].strip(),
    ]

    fail_to_pass = json.loads(instance.get("FAIL_TO_PASS") or "[]")
    if fail_to_pass:
        sections.extend([
            "",
            "Failing tests to target:",
            "\n".join(f"- {item}" for item in fail_to_pass),
        ])

    sections.extend(["", "When you are done, stop without extra commentary."])
    return "\n".join(sections)

def parse_token_stats(output: str) -> dict[str, Any]:
    """Parse opencode JSONL output and aggregate token stats from step_finish events."""
    stats = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0}
    for line in output.splitlines():
        try:
            event = json.loads(line)
            if event.get("type") == "step_finish":
                tokens = event.get("part", {}).get("tokens", {})
                stats["input"] += tokens.get("input", 0)
                stats["output"] += tokens.get("output", 0)
                cache = tokens.get("cache", {})
                stats["cache_read"] += cache.get("read", 0)
                stats["cache_write"] += cache.get("write", 0)
                stats["cost"] += event.get("part", {}).get("cost", 0)
        except json.JSONDecodeError:
            continue
    stats["cost"] = round(stats["cost"], 4)
    return stats


async def capture_patch(runtime: Any) -> str:
    patch_path = "/tmp/model.patch"
    result = await runtime.execute(
        Command(
            command=f"GIT_PAGER=cat git -C /repo --no-pager diff --binary > {patch_path}",
            shell=True,
            timeout=300,
        )
    )
    if result.exit_code not in (0, None):
        raise RuntimeError(f"git diff capture failed: {result.stderr or result.stdout}")
    return (await runtime.read_file(ReadFileRequest(path=patch_path))).content


async def run_instance(
    variant: str,
    image: modal.Image,
    instance: dict[str, Any],
    secret: modal.Secret,
    model: str,
    output_dir: Path,
) -> dict[str, Any]:
    safe_name = instance["instance_id"].replace("/", "__")
    artifact_path = output_dir / variant / f"{safe_name}.json"
    prompt = build_prompt(instance)
    deployment = None

    try:
        deployment = ModalDeployment(
            image=image,
            install_pipx=False,
            startup_timeout=STARTUP_TIMEOUT,
            runtime_timeout=RUNTIME_TIMEOUT,
            deployment_timeout=DEPLOYMENT_TIMEOUT,
            modal_sandbox_kwargs={"secrets": [secret]},
        )
        await deployment.start()
        runtime = deployment.runtime

        await runtime.create_session(CreateBashSessionRequest(session="shell"))
        await runtime.write_file(WriteFileRequest(path="/tmp/problem.txt", content=prompt))

        await runtime.run_in_session(BashAction(
            session="shell",
            command=f"git clone --filter=blob:none --no-checkout https://github.com/{instance['repo']}.git /repo",
            timeout=CLONE_TIMEOUT,
        ))
        await runtime.run_in_session(BashAction(
            session="shell",
            command=f"git -C /repo checkout {instance['base_commit']}",
            timeout=120,
        ))

        opencode_start = time.time()
        opencode_result = await runtime.run_in_session(BashAction(
            session="shell",
            command=f'cd /repo && opencode run --agent build --model {model} --format json "$(cat /tmp/problem.txt)"',
            timeout=COMMAND_TIMEOUT,
            check="silent",
        ))
        duration = round(time.time() - opencode_start, 2)
        diff = await capture_patch(runtime)
        stats = parse_token_stats(opencode_result.output or "")

        save_log(output_dir, variant, instance["instance_id"], opencode_result.output or "")

        payload = {
            "instance_id": instance["instance_id"],
            "variant": variant,
            "model_patch": diff,
            "duration": duration,
            **stats,
        }
        save_artifact(artifact_path, payload)
        return payload
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        save_log(output_dir, variant, instance["instance_id"], f"ERROR:\n{error}")
        return {
            "instance_id": instance["instance_id"],
            "variant": variant,
            "model_patch": "",
            "duration": None,
        }
    finally:
        if deployment:
            try:
                await deployment.stop()
            except Exception:
                pass
            sandbox = getattr(deployment, "sandbox", None)
            if sandbox:
                try:
                    await sandbox.terminate.aio()
                except Exception:
                    pass


async def run_variant(
    variant: str,
    image: modal.Image,
    instances: list[dict[str, Any]],
    secret: modal.Secret,
    model: str,
    concurrency: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_limit(instance: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            print(f"[start] {variant} {instance['instance_id']}")
            result = await run_instance(variant, image, instance, secret, model, output_dir)
            print(f"[done]  {variant} {instance['instance_id']} patch_chars={len(result['model_patch'])}")
            return result

    return await asyncio.gather(*(run_with_limit(inst) for inst in instances))


async def main() -> None:
    split = CONFIG.get("split", "dev")
    output_dir = Path(CONFIG.get("output_dir", "outputs")).resolve()
    concurrency = CONFIG.get("concurrency", 8)
    model = CONFIG.get("model")

    api_key = os.environ.get(API_KEY_ENV_VAR)
    instances = [dict(item) for item in load_dataset(DATASET, split=split)][:1]  # DEBUG: limit to 1
    secret = modal.Secret.from_dict({API_KEY_ENV_VAR: api_key})
    variants = list(VARIANT_IMAGES.keys())

    for variant in variants:
        results = await run_variant(
            variant, VARIANT_IMAGES[variant], instances, secret,
            model, concurrency, output_dir
        )
        predictions_path = save_predictions(output_dir, variant, results)
        print(f"[predictions] {variant}: {predictions_path}")


if __name__ == "__main__":
    asyncio.run(main())
