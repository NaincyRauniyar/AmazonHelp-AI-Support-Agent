# AmazonHelp AI Support Agent — Hiver SDE Intern Take-Home

An AI support agent for **AmazonHelp** (built on the Kaggle "Customer
Support on Twitter" dataset) that:

1. Classifies each incoming customer message into one of 9 intents.
2. Drafts a reply grounded in how AmazonHelp has historically resolved
   similar issues (retrieval over real past resolutions, not a generic
   LLM answer).
3. Decides auto-handle vs. escalate-to-human, with a stated reason for
   every decision.

Built with **zero paid services.** See "Cost" below.

## Why Python, not the stack in my resume

My usual stack is Java/Spring Boot/Kafka/Redis (backend microservices),
but this assignment is fundamentally an NLP/data pipeline problem —
dataset wrangling, embeddings, LLM prompting, evaluation — where Python's
ecosystem (pandas, scikit-learn, sentence-transformers) is the right tool,
not a backend service framework. Happy to talk through how I'd wrap this
as a Spring Boot service behind a REST API if that's useful context.

## Repo layout

```
src/
  config.py          all settings: brand, intents, model/provider choice, paths
  data_prep.py        raw Kaggle CSV -> clean (customer_message, brand_reply) pairs
  retrieval.py         embedding index + nearest-neighbor search over historical pairs
  intents.py            LLM zero-shot classifier + TF-IDF simple-baseline classifier
  llm_client.py          provider-agnostic wrapper (ollama / groq / gemini, all free)
  agent.py                orchestrates classify -> retrieve -> draft -> escalate
  baselines.py              trivial + simple baselines for both sub-tasks
  eval/
    metrics.py               accuracy/F1/precision/recall, judge-human agreement
    llm_judge.py               LLM-as-judge rubric scorer for reply quality
    run_eval.py                 the evaluation harness (run this for headline numbers)
scripts/
  download_data.sh      Kaggle download helper (free account required)
  make_sample_data.py    generates a small bundled demo dataset (no download needed)
  label_golden_set.py     interactive CLI for hand-labeling the golden eval set
  run_pipeline.py           quick end-to-end demo on 3 example messages
data/golden_eval/          golden set + labeling guide
report/REPORT.md            the required report (fill-in-the-numbers template)
decision_log.md               the required decision log
tests/                          unit tests not requiring an LLM or the full dataset
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your env manager of choice
pip install -r requirements.txt
cp .env.example .env
```

### Choose a free LLM provider (edit `.env`)

| Provider | Setup | Cost |
|---|---|---|
| `ollama` (default) | Install from ollama.com, then `ollama pull llama3.1:8b` | Free, fully local, no signup |
| `groq` | Free API key at console.groq.com, set `GROQ_API_KEY` | Free tier (rate-limited) |
| `gemini` | Free API key at aistudio.google.com, set `GEMINI_API_KEY` | Free tier (rate-limited) |

Set `HIVER_LLM_PROVIDER` in `.env` to whichever you use.

## Two ways to run this

### A. Fast demo path (< 5 minutes, no Kaggle account needed)

Uses a small (30-example) hand-written sample dataset bundled in the repo
instead of the full Kaggle download, so you can see the pipeline work
immediately:

```bash
python scripts/make_sample_data.py     # writes data/processed/amazonhelp_threads.parquet
python -m src.retrieval                # builds the embedding index (downloads all-MiniLM-L6-v2, ~90MB, free)
python scripts/run_pipeline.py         # runs the agent on 3 example messages, prints output
```

**This path does not produce the golden-set evaluation or the numbers in
`report/REPORT.md`** — the sample dataset is too small and synthetic for
that. It exists so a reviewer can see real agent behavior fast.

### B. Full reproduction path (real data, real headline results, < 15 min after data download)

```bash
# 1. Get the real dataset (one-time; needs a free Kaggle account)
bash scripts/download_data.sh

# 2. Build the brand-specific thread table (subsampled, see HIVER_MAX_THREADS in .env)
python -m src.data_prep

# 3. Build the retrieval index
python -m src.retrieval

# 4. See it work on a few examples
python scripts/run_pipeline.py

# 5. Build the golden eval set (interactive — this is the ~2-4 hour manual part,
#    not part of the 15-minute reproduction window; a pre-labeled golden set
#    is expected to already be committed at data/golden_eval/golden_eval_labeled.csv
#    for graders re-running this)
python scripts/label_golden_set.py --n 200

# 6. Run the evaluation harness -> headline numbers
python -m src.eval.run_eval --judge-n 40
```

Steps 2-4 and 6 are what should complete in under 15 minutes once the raw
CSV is on disk and `.env` is configured — that's the reproducibility bar
the assignment asks for. Step 5 (hand-labeling) is explicitly excluded
from that window, same as any real eval-set construction would be.

## Get to a submittable report

After running step 6 above, open `report/REPORT.md` and replace every
`[FILL IN]` marker with the real numbers printed by `run_eval.py` (also
saved to `data/processed/eval_results.csv` and
`data/processed/judge_scores.csv` for the failure-analysis section).

## Tests

```bash
python -m pytest tests/ -v
```

These cover the TF-IDF baseline and the retrieval index and don't need an
LLM provider or the real dataset — they run against small in-memory
fixtures.

## Cost

Every component here is free:

- **LLM**: Ollama (fully local) by default; Groq/Gemini free tiers as
  alternatives. No credit card needed for any of the three.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, open weights,
  runs on CPU locally, no API calls.
- **Dataset**: Kaggle's free tier (just needs a free account + API token).
- **Everything else**: pandas/scikit-learn/numpy, all open source.

## What's borrowed

- `sentence-transformers/all-MiniLM-L6-v2` — pretrained open-weights
  embedding model from Hugging Face (Reimers & Gurevych, sentence-BERT
  family), used as-is for retrieval, not fine-tuned.
- Dataset: Kaggle "Customer Support on Twitter"
  (`thoughtvector/customer-support-on-twitter`).
- No other borrowed code (no copied agent-framework boilerplate, no
  copied prompt templates) — the prompts in `src/intents.py`,
  `src/agent.py`, and `src/eval/llm_judge.py` were written for this
  assignment specifically.

## Notes for the live code walkthrough

- `src/agent.py::_decide_action` is the single place all escalation logic
  lives — start there if asked to explain or modify the escalation policy.
- `src/config.py` is the single place all tunables live — brand, intents,
  thresholds, model choice. Nothing else in the codebase should have a
  hardcoded brand name or magic number; if you spot one while reviewing,
  that's a bug, not a design choice.
