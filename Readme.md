# OpenCode SWE-bench Eval

Evaluate OpenCode variants on SWE-bench Lite using Modal for parallel execution.

## Prerequisites

- Python 3.12+
- [Modal](https://modal.com) account
- API key for your model provider (OpenAI, Anthropic, etc.)

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure Modal:
```bash
modal setup
```

3. Set your API key in a `.env`:
```bash
export OPENAI_API_KEY=your_key
```

## Configuration

Create a config file in `configs/` (see `configs/example.yaml`):

```yaml
# Dataset split to evaluate
split: dev

# Where to save results
output_dir: outputs

# Number of parallel instances on Modal
concurrency: 24

# Model to use (provider/model format)
model: openai/gpt-5-mini
api_key_env: OPENAI_API_KEY

# Variants to compare
variants:
  # Official npm release of opencode
  official:
    type: npm

  # Custom fork
  fork-main:
    type: fork
    repo: https://github.com/omkaark/opencode
    branch: main
```

## Usage

### 1. Generate Patches

Run OpenCode on all SWE-bench instances for each variant:

```bash
uv run python src/runner.py --config configs/example.yaml
```

This will:
- Spin up Modal sandboxes in parallel
- Clone each repo at the correct commit
- Run OpenCode to generate patches
- Save artifacts and predictions

### 2. Evaluate & Analyze

Run the SWE-bench harness to test patches and show results:

```bash
uv run python src/analyze.py --config configs/example.yaml --run-harness
```

Or use cached harness results (if already evaluated):

```bash
uv run python src/analyze.py --config configs/example.yaml
```

### Example Output

```
================================================================================
VARIANT ANALYSIS
================================================================================

Variant         Resolved       Rate
-----------------------------------
official            4/23      17.4%
fork-main           4/23      17.4%

Duration (s)
Variant              Avg          P50          P90
-------------------------------------------------
official           87.67        78.57       121.74
fork-main          99.35        86.71       174.24

Input Tokens
Variant              Avg          P50          P90
-------------------------------------------------
official        36654.22      23962.0      76795.2
fork-main       52364.26      40450.0     116628.6
```

## Output Structure

```
outputs/
├── predictions_{variant}.jsonl   # Predictions for SWE-bench harness
└── {variant}/
    ├── {instance_id}.json        # Per-instance artifact (patch, duration, tokens, cost)
    └── logs/{instance_id}.log    # Raw OpenCode output

logs/run_evaluation/{variant}/    # SWE-bench harness results
└── {model}/{instance_id}/
    └── report.json               # Test pass/fail results
```
