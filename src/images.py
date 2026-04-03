import json
from pathlib import Path

import modal

DEFAULT_MODEL = "openai/gpt-5.4"
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
            "openai": {
                "options": {
                    "apiKey": "{env:OPENAI_API_KEY}",
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


def _build_opencode_commands() -> list[str]:
    """Shared build steps for fork-based variants."""
    return [
        "cd /opencode && /root/.bun/bin/bun install --frozen-lockfile || /root/.bun/bin/bun install",
        "cd /opencode/packages/opencode && /root/.bun/bin/bun run build --single",
        "ln -sf /opencode/packages/opencode/dist/opencode-linux-x64/bin/opencode /usr/local/bin/opencode",
    ]


# Supported official baseline from the current OpenCode lineage.
official = _base_image().apt_install("nodejs", "npm").run_commands("npm i -g opencode-ai@latest")


# The fork branch is built into a native Linux binary during the image build.
fork = _base_image().run_commands(
    "curl -fsSL https://bun.sh/install | bash",
    f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    *_build_opencode_commands(),
)


# The tree variant uses a patched fork that reads /tmp/tree_context.json
# (written by the runner) and uses it as compact subagent context instead
# of forking the full conversation prefix.
#
# Build strategy: clone the same fork branch, then overwrite ONLY the
# patched file (tool/task.ts) from the local checkout. This avoids
# uploading the entire 5GB repo with node_modules/.git/dist.
TREE_FORK_DIR = Path(__file__).resolve().parent.parent.parent / "opencode-fork"
PATCHED_TASK_TS = TREE_FORK_DIR / "packages" / "opencode" / "src" / "tool" / "task.ts"
NOSUBAGENT_TASK_TS = TREE_FORK_DIR / "packages" / "opencode" / "src" / "tool" / "task.nosubagent.ts"
NOSUBAGENT_PROMPT = TREE_FORK_DIR / "packages" / "opencode" / "src" / "session" / "prompt" / "default.nosubagent.txt"
SUMMARIZE_TASK_TS = TREE_FORK_DIR / "packages" / "opencode" / "src" / "tool" / "task.summarize.ts"
SUMMARIZE_FORCED_TASK_TS = TREE_FORK_DIR / "packages" / "opencode" / "src" / "tool" / "task.summarize-forced.ts"
SUMMARIZE_RELAXED_PROMPT = TREE_FORK_DIR / "packages" / "opencode" / "src" / "session" / "prompt" / "default.summarize-relaxed.txt"

tree = (
    _base_image()
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    )
    .add_local_file(
        str(PATCHED_TASK_TS),
        "/opencode/packages/opencode/src/tool/task.ts",
        copy=True,
    )
    .run_commands(*_build_opencode_commands())
)


# The "aware" variant uses the same fork branch but with an improved
# subagent system prompt that explicitly tells the subagent it has
# the parent's shared context prefix and should NOT redo work.
# This tests whether the re-verification behavior is prompt-overridable.
aware = (
    _base_image()
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    )
    .add_local_file(
        str(PATCHED_TASK_TS),
        "/opencode/packages/opencode/src/tool/task.ts",
        copy=True,
    )
    .run_commands(*_build_opencode_commands())
)


# The "summarize" variant forces the subagent to first summarize what
# it already knows from the parent's context before proceeding. Tests
# FeDriK's hypothesis that making the model "remember" its context
# prevents re-exploration.
summarize = (
    _base_image()
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    )
    .add_local_file(
        str(SUMMARIZE_TASK_TS),
        "/opencode/packages/opencode/src/tool/task.ts",
        copy=True,
    )
    .run_commands(*_build_opencode_commands())
)


# The "nosubagent" variant removes all "subagent" framing from the
# prompt. The forked agent is told the conversation is its own prior
# work and to continue where it left off. Tests whether the word
# "subagent" triggers re-verification behavior. Task tool is already
# disabled to prevent recursion.
nosubagent = (
    _base_image()
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    )
    .add_local_file(
        str(NOSUBAGENT_TASK_TS),
        "/opencode/packages/opencode/src/tool/task.ts",
        copy=True,
    )
    .add_local_file(
        str(NOSUBAGENT_PROMPT),
        "/opencode/packages/opencode/src/session/prompt/default.txt",
        copy=True,
    )
    .run_commands(*_build_opencode_commands())
)


# "summarize-relaxed" — uses a two-step pipeline that FORCES the model
# to write a summary before acting (first call has tools disabled).
# Also relaxes system prompt constraints. Tests whether explicit summary
# generation helps beyond just the framing.
summarize_relaxed = (
    _base_image()
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        f"git clone --depth 1 --branch {FORK_BRANCH} {FORK_REPO} /opencode",
    )
    .add_local_file(
        str(SUMMARIZE_FORCED_TASK_TS),
        "/opencode/packages/opencode/src/tool/task.ts",
        copy=True,
    )
    .add_local_file(
        str(SUMMARIZE_RELAXED_PROMPT),
        "/opencode/packages/opencode/src/session/prompt/default.txt",
        copy=True,
    )
    .run_commands(*_build_opencode_commands())
)


VARIANT_IMAGES = {
    "official": official,
    "fork": fork,
    "tree": tree,
    "aware": aware,
    "summarize": summarize,
    "summarize-relaxed": summarize_relaxed,
    "nosubagent": nosubagent,
}
