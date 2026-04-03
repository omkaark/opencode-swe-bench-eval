# OpenCode SWE-bench Eval

Evaluating context forking strategies for OpenCode subagents on SWE-Bench. Tests whether giving subagents inherited parent context helps, hurts, or makes no difference — and what prompt strategies improve efficiency.

## Branch: sofia/gpt5-experiments

This branch runs experiments with **GPT-5.4** (OpenAI) instead of the default mimo model. Tests 4 prompt variants to study how context inheritance affects subagent behavior on a competent model.

For full results and analysis, see **[results/README.md](results/README.md)**.

## Quick Results (GPT-5.4, dev split, 23 instances)

| Variant | Resolved | Duration | Tool Calls |
|---|---|---|---|
| **summarize** | **9/21 (42%)** | 126s | 25.7 |
| **summarize-relaxed** | **9/21 (42%)** | 102s | 24.6 |
| official (baseline) | 8/21 (38%) | 89s | 23.5 |
| fork | 6/21 (28%) | 112s | 25.8 |

Summarize variants beat baseline on resolve rate. Fork is consistently worst.

## Repo Structure

```
src/                           # core runner code
  runner.py                    # main OpenCode runner (Modal)
  run_cli_variant.py           # Claude/Codex local runner
  images.py                    # Modal image definitions + variant configs
  tree_builder.py              # structured tree builder
  tree_watcher.py              # live tree watcher (runs in sandbox)

analysis/                      # analysis scripts
  compare_agents.py            # cross-agent comparison table
  compare_variants.py          # n-way variant comparison with CSV
  parse_instance.py            # parse NDJSON into structured summary
  build_tree.py                # build tree from saved artifact

results/                       # eval reports organized by experiment
  README.md                    # full results + analysis ← START HERE
  mimo-dev/                    # mimo model eval reports
  gpt5mini-dev/                # GPT-5.4-mini eval reports
  gpt5full-dev/                # GPT-5.4 full eval reports
  gpt5full-fixed-dev/          # tool-deny fix attempt

outputs/                       # mimo run artifacts (baseline, fork, aware, etc.)
outputs-gpt5/                  # GPT-5.4-mini run artifacts
outputs-gpt5-full/             # GPT-5.4 full run artifacts ← MAIN DATA
outputs-gpt5-full-fixed/       # fixed summarize attempt

logs/                          # per-instance SWE-bench eval logs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install modal swe-rex datasets swebench pydantic aiohttp

modal setup
```

Add your API keys to `.env`:
```
OPENCODE_API_KEY=...
OPENAI_API_KEY=...
```

## Running Experiments

```bash
source .venv/bin/activate
export $(cat .env | xargs)

# run all 4 variants on dev split
python src/runner.py --split dev \
  --variant official --variant fork \
  --variant summarize --variant summarize-relaxed \
  --concurrency 2 --output-dir outputs-gpt5-full

# evaluate results
python -m swebench.harness.run_evaluation \
  -p outputs-gpt5-full/predictions_official.jsonl \
  -s dev -id gpt5full-official --modal True
```

## Variants

| Variant | What it does |
|---|---|
| `official` | Stock OpenCode, subagents start fresh (no forking) |
| `fork` | Subagent gets full parent conversation prefix (~22k tokens) |
| `summarize` | Fork + prompt asks model to summarize context before acting |
| `summarize-relaxed` | Fork + two-step pipeline (first call processes context, second call does work) |
| `nosubagent` | Fork + removes "subagent" framing + edits system prompt (mimo only) |
| `aware` | Fork + explicitly tells subagent about shared context (mimo only) |
| `tree` | Fork + compact structured tree (~2k tokens) instead of raw prefix (mimo only) |

## Reading the Data

Each instance artifact (e.g. `outputs-gpt5-full/official/marshmallow-code__marshmallow-1343.json`) contains:

- `opencode_output` — full NDJSON trace of every tool call, token count, and text output
- `opencode_duration_seconds` — wall-clock time
- `model_patch` — the git diff the agent produced
- `error` — empty if successful

The NDJSON trace has events: `step_start`, `tool_use` (with tool name, input, output), `step_finish` (with token counts including cache.read), `text`.

## Key Options

| Flag | Default | Description |
|---|---|---|
| `--variant` | official, fork | which variant(s) to run |
| `--split` | dev | dataset split (dev has 23, test has 300) |
| `--concurrency` | 2 | parallel Modal sandboxes |
| `--command-timeout` | 1200 | seconds for opencode run |
| `--output-dir` | outputs | where to save artifacts |
| `--skip-existing` | false | reuse successful artifacts |
| `--model` | openai/gpt-5.4 | model in provider/model format |
