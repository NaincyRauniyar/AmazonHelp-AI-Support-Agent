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


