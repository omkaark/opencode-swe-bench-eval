import json

import modal

DEFAULT_MODEL = "opencode/mimo-v2-pro-free"
FORK_BRANCH = "omkaark/subagent-shared-prefix"
FORK_REPO = "https://github.com/omkaark/opencode"

_COMMON_PACKAGES = (
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

_PATH = "/usr/local/bin:/root/.bun/bin:/root/.local/bin:/usr/bin:/bin"

_OPENCODE_CONFIG = json.dumps(
    {
        "$schema": "https://opencode.ai/config.json",
        "model": DEFAULT_MODEL,
        "provider": {
            "opencode": {
                "options": {
                    "apiKey": "{env:OPENCODE_API_KEY}",
                }
            }
        },
    },
)


def _base_image() -> modal.Image:
    return (
        modal.Image.debian_slim()
        .apt_install(*_COMMON_PACKAGES)
        .pip_install("swe-rex==1.4.0")
        .run_commands(
            "ln -sf /usr/bin/fdfind /usr/local/bin/fd || true",
            "mkdir -p /root/.config/opencode",
            f"echo '{_OPENCODE_CONFIG}' > /root/.config/opencode/opencode.json",
        )
        .env({"PATH": _PATH})
    )


# Supported official baseline from the current OpenCode lineage.
official = _base_image().apt_install("nodejs", "npm").run_commands("npm i -g opencode-ai@latest")


# The fork branch is built into a native Linux binary during the image build.
fork = _base_image().run_commands(
    "curl -fsSL https://bun.sh/install | bash",
    f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    "cd /opencode && /root/.bun/bin/bun install --frozen-lockfile || /root/.bun/bin/bun install",
    "cd /opencode/packages/opencode && /root/.bun/bin/bun run build --single",
    "ln -sf /opencode/packages/opencode/dist/opencode-linux-x64/bin/opencode /usr/local/bin/opencode",
)


VARIANT_IMAGES = {
    "official": official,
    "fork": fork,
}
