import json
from pathlib import Path
from typing import Any


def save_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_log(output_dir: Path, variant: str, instance_id: str, content: str) -> Path:
    safe_name = instance_id.replace("/", "__")
    log_path = output_dir / variant / "logs" / f"{safe_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content)
    return log_path


def save_predictions(output_dir: Path, variant: str, results: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictions_{variant}.jsonl"
    with path.open("w") as handle:
        for result in results:
            handle.write(json.dumps({
                "instance_id": result["instance_id"],
                "model_name_or_path": f"opencode-{variant}",
                "model_patch": result["model_patch"],
            }) + "\n")
    return path
