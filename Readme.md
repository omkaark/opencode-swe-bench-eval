# OpenCode SWE-bench Eval

Run SWE-bench evaluations with OpenCode variants on Modal.

## Setup

```bash
uv pip install modal 'swe-rex[modal]' datasets pyyaml
modal setup
export OPENCODE_API_KEY=your_key
```

## Config

Edit `config.yaml`:

```yaml
split: dev
output_dir: outputs
concurrency: 16
model: anthropic/claude-sonnet-4-5-20250929

variants:
  official:
    type: npm
  fork-main:
    type: fork
    repo: https://github.com/omkaark/opencode
    branch: main
```

## Run

```bash
python runner.py
```

## Evaluate

```bash
python -m swebench.harness.run_evaluation \
  --predictions_path outputs/predictions_official.jsonl \
  --swe_bench_tasks princeton-nlp/SWE-bench_Lite \
  --split dev \
  --run_id official
```

## Output

```
outputs/
├── predictions_{variant}.jsonl   # For SWE-bench eval
└── {variant}/
    ├── {instance_id}.json        # Artifact (patch, duration, tokens, cost)
    └── logs/{instance_id}.log    # Raw opencode output
```
