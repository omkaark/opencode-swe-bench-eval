# Experiment Log

## Overview

This repo evaluates context forking strategies for OpenCode subagents on SWE-Bench Lite. We test whether giving subagents the parent's context helps, hurts, or makes no difference — and what prompt strategies can improve efficiency.

## Variants Tested

### Baseline (`official`) — Omkaar
- Stock opencode-ai from npm, no context forking
- Subagents start with a blank slate
- Data: `outputs/full-dev-opencode-mimo-v2-pro-free-c24/official/`

### Fork (`fork`) — Omkaar
- Omkaar's PR (`omkaark/subagent-shared-prefix`)
- Forks full parent conversation prefix (~22k tokens) into subagents
- Task prompt: `"You are a subagent. Proceed with: {task}"`
- Data: `outputs/full-dev-opencode-mimo-v2-pro-free-c24/fork/`

### Aware (`aware`) — Sofia
- Same fork as above but task prompt explicitly tells subagent about shared context
- Task prompt: `"You have the parent's full conversation context. DO NOT re-read files or re-run searches already in your context."`
- System prompt: unchanged (`default.txt`)
- Data: `outputs/aware/`

### Summarize (`summarize`) — Sofia
- Fork + task prompt asks subagent to summarize what it knows before acting
- Task prompt: `"STEP 1: summarize what files have been read, what searches were run, what was learned. STEP 2: proceed with task using that knowledge."`
- System prompt: unchanged (`default.txt`)
- Note: model skips the summary step and goes straight to tools, but still produces better results
- Data: `outputs/summarize/`

### Nosubagent (`nosubagent`) — Sofia
- Fork + removes all "subagent" framing from task prompt + edits system prompt
- Task prompt: `"Continue with the following task. The conversation above is shared context from a prior session."`
- System prompt: patched `default.txt` with "Context sharing" section, changed "use search tools extensively" to "skip to implementation if context already has it"
- Data: `outputs/nosubagent/`

### Tree (`tree`) — Sofia
- Fork + structured tree context (~2k tokens) instead of raw prefix (~22k tokens)
- Tree tracks: files read, edits made, failed searches, successful searches
- Uses `tree_watcher.py` running inside the sandbox to build the tree live
- Data: eval logs at `logs/run_evaluation/tree/`, predictions at `outputs/predictions_tree.jsonl`. Raw trace outputs not available — needs re-run on dev to regenerate.

### Claude Code (`claude`)
- Claude Code (Anthropic), no forking, run locally
- Command: `claude -p <prompt> --output-format json --max-turns 50 --allowedTools Edit,Read,Write,Bash,Glob,Grep`
- Data: `outputs/claude/`

### Codex (`codex`)
- Codex (OpenAI), no forking, run locally
- Command: `codex exec --full-auto --skip-git-repo-check --json <prompt>`
- Data: `outputs/codex/`

## Results (Dev Split, 23 Instances)

```
Variant     | Duration | Tool Calls | Tokens  | Resolved
------------|----------|------------|---------|----------
Summarize   |   145s   |   27.0     |  552k   |  5/20
Baseline    |   181s   |   28.2     |  668k   |  5/22
Nosubagent  |   195s   |   27.3     |  565k   |  5/20
Fork        |   233s   |   33.4     |  874k   |  5/22
Aware       |   243s   |   31.0     |  654k   |  5/22
Claude      |   108s   |   17.6     |  416k   |  7/23
Codex       |   152s   |   13.7     |   n/a   |  7/20
```

Resolved counts are from SWE-Bench eval. Not all agents completed all 23 instances due to broken test environments (astroid-1866, astroid-1978) and Modal sandbox errors.

## Key Findings

1. **Fork adds overhead, not efficiency.** Fork solved the exact same instances as baseline but was slower (14/23 instances), used +5.2 more tool calls, +206k more tokens per instance.

2. **The model redoes parent work despite having it in context.** Traces show fork agents re-running the same greps, re-reading the same files, and re-doing the same git commands that are already in the forked context. See astroid-1333: baseline 473s/64 calls → fork 1124s/69 calls, same edit at the end.

3. **Summarize beats baseline.** 36s faster, fewer tool calls, 17% fewer tokens, same solve rate. The model doesn't actually write a summary (skips to tools) but the prompt framing still reduces redundant calls.

4. **Nosubagent helps but doesn't beat baseline on duration.** Removing "subagent" framing + editing system prompt reduced tool calls and tokens vs fork but duration was still +14s vs baseline.

5. **KV cache is working for fork but not for summarize.** Fork step 1: 10,784 cache_read. Summarize step 1: 1,856 cache_read. Summarize's improvement is purely behavioral.

6. **Claude and Codex (no forking) outperform all OpenCode variants on resolve rate.** 7/23 vs 5/22. But these are different models — the comparison shows the bar, not a direct fork-vs-no-fork comparison.

## How to Read the Data

Each instance artifact (e.g. `outputs/summarize/pylint-dev__astroid-1196.json`) contains:

```json
{
  "variant": "summarize",
  "instance_id": "pylint-dev__astroid-1196",
  "repo": "pylint-dev/astroid",
  "base_commit": "...",
  "opencode_output": "...NDJSON with full trace...",
  "opencode_duration_seconds": 165.3,
  "model_patch": "...git diff...",
  "error": ""
}
```

The `opencode_output` field is NDJSON — one JSON object per line. Key event types:
- `step_start` — new agent step
- `tool_use` — tool call with `part.tool`, `part.state.input`, `part.state.output`
- `step_finish` — token counts in `part.tokens` (total, input, output, cache.read, cache.write)
- `text` — agent's text output

## How to Run a Variant

```bash
source .venv/bin/activate
export $(cat .env | xargs)

# Run on dev split (23 instances, ~1-2 hours)
python src/runner.py --split dev --variant summarize --concurrency 2

# Run on test split (300 instances, ~6-8 hours)
python src/runner.py --split test --variant summarize --concurrency 4

# Evaluate with SWE-Bench
python -m swebench.harness.run_evaluation \
  -p outputs/predictions_summarize.jsonl \
  -s dev -id opencode-summarize --modal True
```

Available variants: `official`, `fork`, `tree`, `aware`, `summarize`, `nosubagent`

## How to Compare Variants

```bash
# Cross-agent comparison (all variants on dev)
python analysis/compare_agents.py

# Parse a single instance into structured summary
python analysis/parse_instance.py outputs/summarize/pylint-dev__astroid-1196.json
```

## File Structure

```
src/                    # Core runners
  runner.py             # Main OpenCode runner (Modal)
  run_cli_variant.py    # Claude/Codex local runner
  images.py             # Modal image definitions + variant configs
  tree_builder.py       # Structured tree builder
  tree_watcher.py       # Live tree watcher (runs in sandbox)

analysis/               # Analysis scripts
  compare_agents.py     # Cross-agent comparison table
  compare_variants.py   # N-way variant comparison with CSV output
  parse_instance.py     # Parse NDJSON into structured summary
  build_tree.py         # Build tree from saved artifact

outputs/                # Run artifacts per variant
eval-reports/           # SWE-Bench eval result JSONs
logs/                   # Per-instance eval logs with report.json
```

## Prompt Variants (in opencode-fork repo)

The task prompt is defined in `packages/opencode/src/tool/task.ts`. Each variant has its own version:
- `task.ts` — current active version
- `task.ts.backup` — original fork prompt
- `task.ts.aware` — aware variant
- `task.nosubagent.ts` — nosubagent variant
- `task.summarize.ts` — summarize variant

The nosubagent variant also patches `packages/opencode/src/session/prompt/default.nosubagent.txt` (replaces `default.txt` in the image build).

## Next Steps

- [ ] Scale summarize to full 300-instance test split
- [ ] Test structured return format (subagent returns JSON with files_changed, diffs, reasoning, failed_approaches)
- [ ] Break down traces by subagent type (explore vs general)
- [ ] Test with system prompt "minimize tokens" constraint relaxed (may allow actual summary generation)
- [ ] Run on SWE-Bench Verified (500 instances) for the paper
