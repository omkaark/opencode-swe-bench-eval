# GPT-5.4 Experiments — sofia/gpt5-experiments

Testing context forking variants with GPT-5.4 on SWE-Bench Lite dev (23 instances).

## What this branch does

Switches the model from mimo-v2-pro-free to OpenAI GPT-5.4 and tests 4 prompt variants to see how a competent model handles inherited subagent context.

## Variants

| Variant | Fork? | Task Prompt | System Prompt |
|---|---|---|---|
| official | No | Stock OpenCode | default.txt |
| fork | Yes | "You are a subagent. Proceed with: {task}" | default.txt |
| summarize | Yes | "Summarize what you know from context first, then proceed" (single call) | default.txt |
| summarize-relaxed | Yes | Two-step pipeline: first call forces context processing, second call does the task | relaxed default.txt |

## Results (GPT-5.4 full)

| Variant | Resolved | Duration | Tool Calls | Tokens |
|---|---|---|---|---|
| **summarize** | **9/21 (42%)** | 126s | 25.7 | 489k |
| **summarize-relaxed** | **9/21 (42%)** | 102s | 24.6 | 400k |
| official | 8/21 (38%) | 89s | 23.5 | 399k |
| fork | 6/21 (28%) | 112s | 25.8 | 445k |

Both summarize variants beat baseline. Fork is worst.

## What I found

### Fork adds overhead without improving accuracy
Fork does +2.3 more tool calls and +46k more tokens than baseline but resolves fewer instances (6 vs 8). Traces show the forked agent re-reads files and re-runs searches already in its context. The inherited context creates noise rather than helping.

### Summarize makes the model more focused
Summarize solved astroid-1196 with 21 tool calls where baseline used 38. It skipped deep dives into inference.py and helpers.py that were irrelevant to the fix. The "summarize first" framing makes the model more surgical.

### The summary step doesn't actually work
The model never writes a summary despite being asked to. It skips straight to tool calls on all 23 instances. The system prompt's "minimize output tokens" instruction suppresses it. I tried:
1. Relaxing the system prompt constraints — model still skips
2. Two-step pipeline with tools disabled on first call — tools:{} means "no changes" not "no tools" in OpenCode (session/prompt.ts line 172)
3. Explicit tool denies (read:false, edit:false, etc.) — first call's output still not captured in NDJSON

Despite the summary not being visible, the two-step pipeline still changes behavior — the first SessionPrompt.prompt() call processes context internally, which affects the second call's planning.

### The explicit tool-deny fix hurt
Ran a separate experiment with all tools explicitly denied on the first call. This dropped resolve rate from 9/21 to 7/21. The explicit denies likely changed session permission state in unintended ways.

## Instance-level analysis

### Instances only summarize variants solved
- **astroid-1196**: summarize did it in 21 tools (6 greps, 11 reads). Official used 38 tools (14 greps, 15 reads) and failed — wasted time exploring irrelevant inference machinery.
- **sqlfluff-1517**: sum-relaxed used 54 tools (most of any variant) but found the right fix through more aggressive testing (15 bash commands vs official's 5).
- **sqlfluff-1625**: sum-relaxed was fastest (85s, 21 tools) with smallest correct patch (1145 chars).

### Instances fork lost
- **astroid-1333**: fork did 47 tools, official did 44. Nearly identical exploration but fork's extra re-reads led to a slightly different (wrong) edit.
- **sqlfluff-1733**: fork only did 59 tools vs official's 90 — inherited context created false confidence, stopped exploring too early.

### Union of all variants
11/21 (52%) — each variant solves instances others miss. Combining approaches has significant headroom.

## GPT-5.4-mini results (for reference)

Ran the same 4 variants on GPT-5.4-mini first. Results were similar to mimo — mini is too weak for meaningful signal. See `gpt5mini-dev/` for data.

| Variant | Resolved |
|---|---|
| summarize-relaxed | 7/21 (33%) |
| official | 6/21 (29%) |
| fork | 5/20 (25%) |
| summarize | 4/21 (19%) |

## Output directories

| Directory | Model | Contents |
|---|---|---|
| outputs-gpt5-full/ | GPT-5.4 | Main results — all 4 variants, 23 instances each |
| outputs-gpt5-full-fixed/ | GPT-5.4 | Summarize-relaxed with tool-deny fix (hurt performance) |
| outputs-gpt5/ | GPT-5.4-mini | All 4 variants, 23 instances each |
| outputs/ | mimo | Earlier experiments (baseline, fork, aware, nosubagent, summarize, claude, codex) |

## Open issues

1. **Summary pipeline not working** — need to debug how SessionPrompt.prompt() handles multiple calls within a task tool. The first call runs internally but output isn't captured in NDJSON.
2. **Only 23 dev instances** — need to scale to SWE-Bench Lite (300) or Verified (500) for statistical power.
3. **Non-determinism** — 1-2 instance differences on 23 tasks could be noise. Larger dataset needed.

## How to reproduce

```bash
source .venv/bin/activate
export $(cat .env | xargs)

# run all 4 variants
python src/runner.py --split dev --variant official --variant fork --variant summarize --variant summarize-relaxed --concurrency 2 --output-dir outputs-gpt5-full

# evaluate
python -m swebench.harness.run_evaluation -p outputs-gpt5-full/predictions_official.jsonl -s dev -id gpt5full-official --modal True
# (repeat for fork, summarize, summarize-relaxed)
```
