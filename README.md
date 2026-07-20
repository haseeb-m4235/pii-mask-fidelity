# pii-mask-fidelity

How many PII masks can an LLM juggle before it drops one?

This project measures how faithfully Claude models reproduce PII masks as the number of masks in context grows.
The motivation is a privacy-preserving agent architecture: before any LLM call, real PII (emails, phone numbers, IPs, credit cards, SSNs) is replaced by masks of the form `[PII_n]`, and after the call a deterministic reverse scrubber substitutes the real values back into the model's answers and tool calls.
That architecture is only safe if the model reproduces masks exactly.
This benchmark quantifies where that assumption breaks, per model, as a function of how many masks are in context.

## Dataset

The dataset is frozen and included in the repo.
All PII is synthetic (faker-generated), so nothing here is sensitive.

- 5 documents of templated business content (support tickets, incident reports, CRM notes, billing operations).
- Each document has 10 pages delimited by `=== PAGE n ===` lines.
- Each document contains exactly 1000 unique PII values: 100 per page, 20 each of email, phone, IPv4, credit card, and SSN.
- Scrubbed copies replace each value with `[PII_n]`, numbered in order of first appearance, so page 1 holds roughly masks 1-100 and page 10 roughly masks 901-1000.
- Every document has 100 evaluation questions (10 per page): 5 answered in prose and 5 requiring a tool call, each targeting exactly one mask with a provably unique correct answer.

Layout:

```
original_docs/    doc_1.txt .. doc_5.txt          the raw documents (synthetic PII)
scrubbed_docs/    doc_1.txt .. doc_5.txt          masked copies fed to the models
  scrubbed_keys/  doc_1.json .. doc_5.json        mask -> real value mapping per doc
prompts/          doc_1.json .. doc_5.json        questions per page with expected masks
scripts/          run_experiment.py               the experiment runner
                  config.py                       run settings (models, docs, pages, ...)
                  constants.py                    paths, system prompt, tools, mask regexes
results/          raw/*.jsonl, summary.csv        created by the runner
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running the experiment

The runner takes no command-line arguments.
Every run setting is a constant in `scripts/config.py`, with its use cases documented inline; edit that file, then start the run:

```bash
.venv/bin/python scripts/run_experiment.py
```

With the default configuration this is the full sweep (3 models x 5 docs x page counts 1-10 = 8250 calls).
Invalid settings (unknown documents, out-of-range page counts, or setting both options of a mutually exclusive pair) exit with an error before any API call.

Smoke test (5 questions against Haiku, effectively free):

```python
MODELS = ["haiku"]
DOCS = [1]
PAGES = [1]
LIMIT = 5
```

Pilot run (2 documents, page counts 1-5, both cheaper models):

```python
MODELS = ["haiku", "sonnet"]
NUM_DOCS = 2
MAX_PAGES = 5
```

Settings:

| Setting | Meaning |
|---|---|
| `MODELS` | Full model ids or shorthands `haiku`, `sonnet`, `opus` (default: all three). |
| `DOCS` / `NUM_DOCS` | Explicit document numbers, or the first N documents. Set at most one; both `None` runs all documents. |
| `PAGES` / `MAX_PAGES` | Explicit page counts (context = pages 1..K), or every page count from 1 to K. Set at most one; both `None` runs 1-10. |
| `LIMIT` | Only the first N questions per cell, for testing (0 = all). |
| `CONCURRENCY` | Parallel requests per cell (default 8). |
| `OVERWRITE` | Re-run cells whose raw output file already exists. |

How a run works: for every (model, document, page count) cell, the system prompt contains a fixed instruction plus pages 1..K of the scrubbed document, and every eligible question is sent as its own fresh single-turn conversation.
The system prompt is served through Anthropic prompt caching (one cache write per cell, then cheap reads), which cuts input cost by roughly 90%.
Progress is shown as a single tqdm bar over every question in the run, with the current cell name as a postfix; the only printed output is the end-of-run summary (output locations plus an accuracy table by model and page count).
Completed cells are skipped on re-runs unless `OVERWRITE = True`, so an interrupted sweep resumes for free.

## Scoring

Scoring is fully deterministic; no LLM judge is involved.
A prose answer is correct iff it contains the expected mask.
A tool-call answer is correct iff the expected tool is called with the exact mask as its argument.
Every wrong answer is classified:

| Class | Meaning |
|---|---|
| `wrong_existing_mask` | A real mask, but the wrong one. The dangerous silent failure: reverse scrubbing succeeds and the agent acts on the wrong person's data. |
| `nonexistent_mask` | A well-formed mask whose index is not in the mapping. Detectable; a reverse scrubber can fail closed. |
| `malformed_mask` | Near-miss forms such as `PII 12`, `[PII-12]`, `[PII_012]`. Detectable. |
| `no_mask` | No mask produced at all. |
| `correct_mask_wrong_usage` | Right mask, wrong delivery: wrong tool, extra text in the argument, or prose instead of a tool call. |
| `api_error` | The call failed after retries. |

When comparing models, look at `wrong_existing_mask` separately from overall accuracy.
A model with lower accuracy but zero silent failures is safer for this architecture than a more accurate one whose errors are all silent.

## Results and analysis

`results/raw/{model}_doc{n}_pages{k}.jsonl` holds one self-describing record per question: model, doc, page count, source page, mode, PII type, expected mask, response text, tool calls, correctness, classification, spurious masks, latency, and token usage.
These files accumulate across runs and are the durable source of truth.
`results/summary.csv` is a per-cell aggregate (accuracy plus every failure-class rate) rewritten on each invocation.

Accuracy versus number of PII in context:

```python
import pandas as pd, glob, json

df = pd.DataFrame(json.loads(line) for f in glob.glob("results/raw/*.jsonl") for line in open(f))
df["n_pii"] = df.page_count * 100
df.groupby(["model", "n_pii", "mode"]).correct.mean().unstack(["model", "mode"]).plot()
```

The raw records also support deeper cuts without re-running anything: accuracy versus mask index, definition-depth (lost-in-the-middle) effects, per-PII-type breakdowns, and failure-class curves.

## Cost estimates

Estimated API cost per model for a full page sweep (page counts 1-10) over the first N documents, with prompt caching active.

| Documents | Questions per model | Haiku 4.5 | Sonnet 5 | Opus 4.8 |
|---|---|---|---|---|
| 1 | 550 | $1.18 | $3.54 | $5.91 |
| 2 | 1100 | $2.36 | $7.08 | $11.81 |
| 3 | 1650 | $3.54 | $10.62 | $17.71 |
| 4 | 2200 | $4.72 | $14.17 | $23.61 |
| 5 | 2750 | $5.90 | $17.71 | $29.52 |

Assumptions: token counts measured from the actual repo files, one cache write per cell, ~80 output tokens per answer, no extended thinking.
Prices as of July 2026 per million tokens (input/output): Haiku 4.5 $1/$5, Sonnet 5 $3/$15, Opus 4.8 $5/$25; cache writes cost 1.25x input, cache reads 0.1x.
If adaptive thinking engages on Sonnet or Opus, output grows by a few hundred tokens per answer; budget roughly 1.5x-2x the listed figure as a ceiling.
Without prompt caching everything costs about 7x more, so if you modify the runner, keep the cache warm-up behavior intact.
