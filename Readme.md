# OpenCode SWE-bench Eval

Run SWE-bench evaluations comparing OpenCode variants using Modal sandboxes.

## Variants

- `official`: Current OpenCode CLI from `npm i -g opencode-ai@latest`
- `fork`: `omkaark/opencode` branch `omkaark/subagent-shared-prefix`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install modal swe-rex datasets swebench pydantic aiohttp

modal setup
export OPENCODE_ZEN_API_KEY=...
```

## Usage

### Smoke test

```bash
python runner.py --smoke-only
```

### Run evaluation

```bash
# Both variants, full dev split
python runner.py --split dev --concurrency 4 --command-timeout 1800

# Single variant
python runner.py --variant fork --split dev --command-timeout 1800

# Single instance
python runner.py --variant fork --instance-id sqlfluff__sqlfluff-1625 --command-timeout 1800
```

### Score results

```bash
python -m swebench.harness.run_evaluation \
  -d princeton-nlp/SWE-bench_Lite \
  -s dev \
  -p outputs/predictions_fork.jsonl \
  -id fork \
  --modal true
```

## Output Structure

```
outputs/
├── predictions_official.jsonl    # SWE-bench predictions file
├── predictions_fork.jsonl
├── official/
│   ├── logs/
│   │   └── {instance_id}.log     # Full opencode output per run
│   └── {instance_id}.json        # Artifact with patch, status, metadata
└── fork/
    ├── logs/
    │   └── {instance_id}.log
    └── {instance_id}.json
```

## Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | official, fork | Which variant(s) to run |
| `--split` | dev | Dataset split (dev, test) |
| `--concurrency` | 2 | Parallel Modal sandboxes |
| `--command-timeout` | 1200 | Seconds for opencode run |
| `--skip-existing` | false | Reuse successful artifacts |
| `--include-hints` | false | Include SWE-bench hints in prompt |
