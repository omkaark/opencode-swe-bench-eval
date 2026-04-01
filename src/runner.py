import argparse
import asyncio
import json
import os
import random
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import modal
from datasets import load_dataset
from pydantic import BaseModel
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import BashAction, Command, CreateBashSessionRequest, ReadFileRequest, WriteFileRequest
from swerex.runtime.remote import RemoteRuntime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from images import DEFAULT_MODEL, VARIANT_IMAGES
from tree_builder import TreeBuilder


def patch_remote_runtime_timeout_handling() -> None:
    if getattr(RemoteRuntime, "_opencode_request_timeout_patch", False):
        return

    async def _request_with_timeouts(
        self: RemoteRuntime,
        endpoint: str,
        payload: BaseModel | None,
        output_class: Any,
        num_retries: int = 0,
    ):
        request_url = f"{self._api_url}/{endpoint}"
        request_id = str(uuid.uuid4())
        headers = self._headers.copy()
        headers["X-Request-ID"] = request_id

        retry_count = 0
        last_exception: Exception | None = None
        retry_delay = 0.1
        backoff_max = 5

        request_timeout = self._get_timeout()
        for field_name in ("timeout", "startup_timeout"):
            value = getattr(payload, field_name, None)
            if value is not None:
                request_timeout = max(request_timeout, float(value) + 60.0)

        client_timeout = aiohttp.ClientTimeout(total=request_timeout)

        while retry_count <= num_retries:
            try:
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
                    async with session.post(
                        request_url,
                        json=payload.model_dump() if payload else None,
                        headers=headers,
                        timeout=client_timeout,
                    ) as resp:
                        await self._handle_response_errors(resp)
                        return output_class(**await resp.json())
            except Exception as exc:
                last_exception = exc
                retry_count += 1
                if retry_count <= num_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    retry_delay += random.uniform(0, 0.5)
                    retry_delay = min(retry_delay, backoff_max)
                    continue
                self.logger.error("Error making request %s after %d retries: %s", request_id, num_retries, exc)
        raise last_exception  # type: ignore[misc]

    RemoteRuntime._request = _request_with_timeouts  # type: ignore[method-assign]
    RemoteRuntime._opencode_request_timeout_patch = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench patch generation with OpenCode via SWE-ReX on Modal.")
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant to run. Repeatable. Defaults to official and fork.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="Specific SWE-bench instance_id to run. Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit after filtering.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for predictions and per-instance artifacts.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per instance after the first attempt.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent Modal sandboxes per variant.",
    )
    parser.add_argument(
        "--clone-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for repository clone commands.",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds for the OpenCode run.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180,
        help="Seconds to wait for the SWE-ReX runtime to become healthy.",
    )
    parser.add_argument(
        "--runtime-timeout",
        type=int,
        default=1800,
        help="Client-side runtime timeout for SWE-ReX actions.",
    )
    parser.add_argument(
        "--deployment-timeout",
        type=int,
        default=3600,
        help="Modal sandbox lifetime timeout.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenCode model in provider/model format.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default="OPENCODE_API_KEY",
        help="Environment variable that holds the API key.",
    )
    parser.add_argument(
        "--include-hints",
        action="store_true",
        help="Include SWE-bench hints_text in the prompt.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-instance artifacts from --output-dir.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the preflight smoke test for each requested variant.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run variant smoke tests and exit without benchmark instances.",
    )
    parser.add_argument(
        "--modal-output",
        action="store_true",
        help="Enable Modal build/runtime logs in the local console.",
    )
    return parser.parse_args()


def get_variants(args: argparse.Namespace) -> list[str]:
    variants = args.variants or ["official", "fork"]
    unknown = [name for name in variants if name not in VARIANT_IMAGES]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}. Available: {', '.join(sorted(VARIANT_IMAGES))}")
    return variants


def require_api_key(env_var: str) -> str:
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key
    raise SystemExit(
        f"Missing API key. Set {env_var} in your environment before running this script."
    )


def parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def build_prompt(instance: dict[str, Any], *, include_hints: bool) -> str:
    sections = [
        "You are working inside the root of a repository.",
        "Fix the issue described below by editing the repository in place.",
        "Keep changes focused, and run targeted checks only if they are practical in this environment.",
        "",
        "Issue:",
        instance["problem_statement"].strip(),
    ]

    fail_to_pass = parse_json_list(instance.get("FAIL_TO_PASS"))
    if fail_to_pass:
        sections.extend(
            [
                "",
                "Failing tests to target:",
                "\n".join(f"- {item}" for item in fail_to_pass),
            ]
        )

    if include_hints:
        hints = (instance.get("hints_text") or "").strip()
        if hints:
            sections.extend(["", "Additional benchmark hint:", hints])

    sections.extend(
        [
            "",
            "When you are done, stop without extra commentary.",
        ]
    )
    return "\n".join(sections)


def load_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = load_dataset(args.dataset, split=args.split)
    items = [dict(item) for item in dataset]

    if args.instance_ids:
        wanted = set(args.instance_ids)
        items = [item for item in items if item["instance_id"] in wanted]

    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        raise SystemExit("No benchmark instances matched the requested filters.")

    return items


def instance_artifact_path(output_dir: Path, variant: str, instance_id: str) -> Path:
    safe_name = instance_id.replace("/", "__")
    return output_dir / variant / f"{safe_name}.json"


def save_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_live_tree(instance: dict[str, Any], variant: str, opencode_output: str) -> TreeBuilder:
    """Build a TreeBuilder from captured opencode NDJSON output."""
    builder = TreeBuilder(
        instance_id=instance["instance_id"],
        variant=variant,
        repo=instance["repo"],
    )
    for line in opencode_output.split("\n"):
        builder.feed_line(line)
    return builder


def save_tree(output_dir: Path, variant: str, instance_id: str, tree: dict[str, Any]) -> Path:
    safe_name = instance_id.replace("/", "__")
    tree_path = output_dir / variant / f"{safe_name}_tree.json"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(json.dumps(tree, indent=2) + "\n")
    return tree_path


def save_log(output_dir: Path, variant: str, instance_id: str, content: str) -> Path:
    safe_name = instance_id.replace("/", "__")
    log_path = output_dir / variant / "logs" / f"{safe_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content)
    return log_path


def load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


async def stop_deployment(deployment: ModalDeployment | None) -> None:
    if deployment is None:
        return

    sandbox = None
    try:
        sandbox = deployment.sandbox
    except Exception:
        sandbox = None

    try:
        await deployment.stop()
    except Exception:
        pass

    # SWE-ReX 1.4.0 only terminates sandboxes after they have already exited.
    if sandbox is not None:
        try:
            if await sandbox.poll.aio() is None:
                await sandbox.terminate.aio()
        except Exception:
            pass


async def bash(runtime: Any, session: str, command: str, *, timeout: int | None = None, check: str = "raise") -> Any:
    return await runtime.run_in_session(
        BashAction(
            session=session,
            command=command,
            timeout=timeout,
            check=check,
        )
    )


def summarize_bash_result(result: Any, *, limit: int = 1200) -> str:
    parts = [f"exit_code={getattr(result, 'exit_code', None)}"]
    for field in ("output", "error"):
        value = getattr(result, field, "") or ""
        if not value:
            continue
        text = value.strip()
        if len(text) > limit:
            text = text[:limit] + "...<truncated>"
        parts.append(f"{field}={text!r}")
    return ", ".join(parts)


async def capture_patch(runtime: Any, *, timeout: int) -> str:
    patch_path = "/tmp/model.patch"
    result = await runtime.execute(
        Command(
            command=f"GIT_PAGER=cat git -C /repo --no-pager diff --binary > {patch_path}",
            shell=True,
            timeout=timeout,
        )
    )
    if result.exit_code not in (0, None):
        raise RuntimeError(
            f"git diff capture failed with exit code {result.exit_code}: {result.stderr or result.stdout}"
        )
    return (await runtime.read_file(ReadFileRequest(path=patch_path))).content


async def capture_git_status(runtime: Any, *, timeout: int) -> str:
    result = await runtime.execute(
        Command(
            command=["git", "-C", "/repo", "status", "--short"],
            timeout=timeout,
        )
    )
    if result.exit_code not in (0, None):
        raise RuntimeError(
            f"git status failed with exit code {result.exit_code}: {result.stderr or result.stdout}"
        )
    return result.stdout


async def smoke_test(
    variant: str,
    image: modal.Image,
    secret: modal.Secret,
    args: argparse.Namespace,
) -> None:
    deployment = ModalDeployment(
        image=image,
        install_pipx=False,
        startup_timeout=args.startup_timeout,
        runtime_timeout=args.runtime_timeout,
        deployment_timeout=args.deployment_timeout,
        modal_sandbox_kwargs={"secrets": [secret]},
    )

    try:
        await deployment.start()
        runtime = deployment.runtime
        await runtime.create_session(CreateBashSessionRequest(session="smoke"))

        version = await bash(runtime, "smoke", "opencode --version", timeout=120, check="silent")
        agents = await bash(runtime, "smoke", "opencode agent list", timeout=120, check="silent")
        ready = await bash(
            runtime,
            "smoke",
            f'opencode run --agent build --model {args.model} --format json "Reply with READY. Do not use any tools."',
            timeout=300,
            check="silent",
        )

        if version.exit_code not in (0, None):
            raise RuntimeError(f"{variant}: opencode --version failed ({summarize_bash_result(version)})")
        if agents.exit_code not in (0, None):
            raise RuntimeError(f"{variant}: opencode agent list failed ({summarize_bash_result(agents)})")
        if ready.exit_code not in (0, None) or "READY" not in ready.output:
            raise RuntimeError(f"{variant}: smoke run did not return READY ({summarize_bash_result(ready)})")

        print(f"[smoke] {variant}: OK")
    finally:
        await stop_deployment(deployment)


async def run_instance(
    variant: str,
    image: modal.Image,
    instance: dict[str, Any],
    secret: modal.Secret,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    artifact_path = instance_artifact_path(output_dir, variant, instance["instance_id"])
    if args.skip_existing and artifact_path.exists():
        existing = load_artifact(artifact_path)
        if not existing.get("error"):
            return existing

    prompt = build_prompt(instance, include_hints=args.include_hints)

    for attempt in range(args.retries + 1):
        deployment = None
        modal_log_url = ""
        try:
            deployment = ModalDeployment(
                image=image,
                install_pipx=False,
                startup_timeout=args.startup_timeout,
                runtime_timeout=args.runtime_timeout,
                deployment_timeout=args.deployment_timeout,
                modal_sandbox_kwargs={"secrets": [secret]},
            )
            await deployment.start()
            modal_log_url = await deployment.get_modal_log_url()
            runtime = deployment.runtime

            await runtime.create_session(CreateBashSessionRequest(session="shell"))
            await runtime.write_file(WriteFileRequest(path="/tmp/problem.txt", content=prompt))

            await bash(
                runtime,
                "shell",
                f"git clone --filter=blob:none --no-checkout https://github.com/{instance['repo']}.git /repo",
                timeout=args.clone_timeout,
            )
            await bash(
                runtime,
                "shell",
                f"git -C /repo checkout {instance['base_commit']}",
                timeout=120,
            )

            # For the tree variant, write the tree context file to the sandbox.
            # The patched fork reads /tmp/tree_context.json and uses it as
            # compact subagent context instead of the raw conversation prefix.
            if variant == "tree":
                # Upload the tree watcher script and start it in the background.
                # It tails the NDJSON output file and continuously updates
                # /tmp/tree_context.json so subagents get a live snapshot of
                # what the parent has learned up to the point they're spawned.
                watcher_src = Path(__file__).with_name("tree_watcher.py").read_text()
                await runtime.write_file(WriteFileRequest(
                    path="/tmp/tree_watcher.py", content=watcher_src,
                ))
                await bash(
                    runtime, "shell",
                    "touch /tmp/opencode_output.ndjson && "
                    "python3 /tmp/tree_watcher.py /tmp/opencode_output.ndjson &",
                    timeout=30,
                    check="silent",
                )

            opencode_start = time.time()
            if variant == "tree":
                # Pipe opencode output through tee so the watcher can tail it
                opencode_cmd = (
                    "cd /repo && "
                    f'opencode run --agent build --model {args.model} --format json '
                    f'"$(cat /tmp/problem.txt)" '
                    f'2>&1 | tee /tmp/opencode_output.ndjson'
                )
            else:
                opencode_cmd = (
                    "cd /repo && "
                    f'opencode run --agent build --model {args.model} --format json "$(cat /tmp/problem.txt)"'
                )
            opencode_result = await bash(
                runtime,
                "shell",
                opencode_cmd,
                timeout=args.command_timeout,
                check="silent",
            )
            opencode_duration_seconds = time.time() - opencode_start

            if variant == "tree":
                # Signal the watcher to stop
                await bash(runtime, "shell", "touch /tmp/opencode_done", timeout=10, check="silent")
            diff = await capture_patch(runtime, timeout=max(300, args.command_timeout))
            status = await capture_git_status(runtime, timeout=120)

            opencode_output = opencode_result.output or ""
            save_log(output_dir, variant, instance["instance_id"], opencode_output)

            # Build and save tree from this run's output
            tree_builder = build_live_tree(instance, variant, opencode_output)
            tree = tree_builder.to_dict()
            tree["total_duration_seconds"] = round(opencode_duration_seconds, 2)
            tree["patch_non_empty"] = bool(diff.strip())
            save_tree(output_dir, variant, instance["instance_id"], tree)

            payload = {
                "variant": variant,
                "instance_id": instance["instance_id"],
                "repo": instance["repo"],
                "base_commit": instance["base_commit"],
                "attempt": attempt + 1,
                "modal_log_url": modal_log_url,
                "opencode_exit_code": opencode_result.exit_code,
                "opencode_output": opencode_output,
                "opencode_duration_seconds": round(opencode_duration_seconds, 2),
                "git_status": status,
                "model_patch": diff,
                "error": "",
            }
            save_artifact(artifact_path, payload)
            return payload
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            if attempt >= args.retries:
                payload = {
                    "variant": variant,
                    "instance_id": instance["instance_id"],
                    "repo": instance["repo"],
                    "base_commit": instance["base_commit"],
                    "attempt": attempt + 1,
                    "modal_log_url": modal_log_url,
                    "opencode_exit_code": None,
                    "opencode_output": "",
                    "opencode_duration_seconds": None,
                    "git_status": "",
                    "model_patch": "",
                    "error": error,
                }
                save_artifact(artifact_path, payload)
                save_log(output_dir, variant, instance["instance_id"], f"ERROR:\n{error}")
                return payload
            print(f"[retry] {variant} {instance['instance_id']} attempt {attempt + 1}: {exc}")
            await asyncio.sleep(min(30, 5 * (attempt + 1)))
        finally:
            await stop_deployment(deployment)

    raise RuntimeError("unreachable")


async def run_variant(
    variant: str,
    image: modal.Image,
    instances: list[dict[str, Any]],
    secret: modal.Secret,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_with_limit(instance: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            print(f"[start] {variant} {instance['instance_id']}")
            result = await run_instance(variant, image, instance, secret, args, output_dir)
            patch_size = len(result["model_patch"])
            print(f"[done]  {variant} {instance['instance_id']} patch_chars={patch_size}")
            return result

    return await asyncio.gather(*(run_with_limit(instance) for instance in instances))


def write_predictions(output_dir: Path, variant: str, results: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictions_{variant}.jsonl"
    with path.open("w") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "instance_id": result["instance_id"],
                        "model_name_or_path": f"opencode-{variant}",
                        "model_patch": result["model_patch"],
                    }
                )
                + "\n"
            )
    return path


async def main() -> None:
    patch_remote_runtime_timeout_handling()
    args = parse_args()
    if args.modal_output:
        modal.enable_output()

    variants = get_variants(args)
    api_key = require_api_key(args.api_key_env_var)
    instances = [] if args.smoke_only else load_instances(args)

    output_dir = Path(args.output_dir).resolve()
    secret_payload = {
        "OPENCODE_API_KEY": api_key,
    }
    if args.api_key_env_var != "OPENCODE_API_KEY":
        secret_payload[args.api_key_env_var] = api_key
    secret = modal.Secret.from_dict(secret_payload)

    if not args.skip_smoke:
        for variant in variants:
            await smoke_test(variant, VARIANT_IMAGES[variant], secret, args)

    if args.smoke_only:
        return

    for variant in variants:
        results = await run_variant(variant, VARIANT_IMAGES[variant], instances, secret, args, output_dir)
        predictions_path = write_predictions(output_dir, variant, results)
        print(f"[predictions] {variant}: {predictions_path}")


if __name__ == "__main__":
    asyncio.run(main())
