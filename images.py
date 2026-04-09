import json
from pathlib import Path

import modal
import yaml

COMMON_PACKAGES = (
    "bash",
    "git",
    "curl",
    "ca-certificates",
    "unzip",
    "zip",
    "ripgrep",
    "fzf",
    "fd-find",
)

PATH = "/usr/local/bin:/root/.bun/bin:/root/.local/bin:/usr/bin:/bin"

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_opencode_config(model: str) -> str:
    return json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": model,
        "provider": {
            "opencode": {
                "options": {
                    "apiKey": "{env:OPENCODE_API_KEY}",
                }
            }
        },
    })


def get_base_image(model: str) -> modal.Image:
    return (
        modal.Image.debian_slim()
        .apt_install(*COMMON_PACKAGES)
        .pip_install("swe-rex[modal]==1.4.0")
        .run_commands(
            "ln -sf /usr/bin/fdfind /usr/local/bin/fd || true",
            "mkdir -p /root/.config/opencode",
            f"echo '{get_opencode_config(model)}' > /root/.config/opencode/opencode.json",
        )
        .env({"PATH": PATH})
    )


def build_npm_image(model: str) -> modal.Image:
    return get_base_image(model).apt_install("nodejs", "npm").run_commands("npm i -g opencode-ai@latest")


def build_fork_image(model: str, repo: str, branch: str) -> modal.Image:
    return get_base_image(model).run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        f"git clone --depth 1 --branch {branch} {repo} /opencode",
        "cd /opencode && /root/.bun/bin/bun install --frozen-lockfile || /root/.bun/bin/bun install",
        "cd /opencode/packages/opencode && /root/.bun/bin/bun run build --single",
        "ln -sf /opencode/packages/opencode/dist/opencode-linux-x64/bin/opencode /usr/local/bin/opencode",
    )


def load_variants(config: dict) -> dict[str, modal.Image]:
    model = config.get("model")
    images = {}
    for name, spec in config.get("variants", {}).items():
        variant_type = spec.get("type", "npm")
        if variant_type == "npm":
            images[name] = build_npm_image(model)
        elif variant_type == "fork":
            repo = spec.get("repo", "https://github.com/omkaark/opencode")
            branch = spec.get("branch", "main")
            images[name] = build_fork_image(model, repo, branch)
        else:
            raise ValueError(f"Unknown variant type: {variant_type}")
    return images


CONFIG = load_config()
VARIANT_IMAGES = load_variants(CONFIG)
