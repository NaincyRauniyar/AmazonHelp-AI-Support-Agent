# Golden Eval Set — Labeling Guide

This describes how `data/golden_eval/golden_eval_labeled.csv` was built.
Read this alongside `scripts/label_golden_set.py`, which is the actual
labeling tool.

## Sampling

- Source: the reconstructed `<brand>_threads.parquet` table (real customer
  message -> real historical brand reply pairs), **not** synthetic data.
- Size: 200 examples (within the required 150-250 range).
- Stratified by message length (short / medium / long, ~1/3 each) rather
  than by intent, because we don't have ground-truth intents to stratify
  by *before* labeling — that would be circular. Length is a cheap, free
  proxy that happens to correlate with message complexity: short messages
  tend to be simple one-issue asks, long messages tend to be rambling
  multi-issue complaints, which is exactly the kind of case mix we want
  the golden set to cover.
- Deliberately NOT filtered to "clean" examples. Some sampled rows are
  genuinely ambiguous (multiple issues in one message, sarcasm, non-English
  fragments) — we labeled them as best we could and left a note rather than
  discarding them, because throwing out hard examples would make our
  reported accuracy look better than the system actually performs in
  production.

## Labeling process

One person labeled all 200 examples for:

- `true_intent` — one of the 9 intents in `src/config.py`. When a message
  plausibly fit two intents, we picked whichever the *customer's primary
  ask* was, and noted the ambiguity in `notes`.
- `true_action` — `auto_handle` if a competent human agent could send a
  reply immediately with no extra info/approval needed; `escalate`
  otherwise. Anything involving account security, an angry/repeat-contact
  customer, or missing information needed to resolve it was marked
  `escalate`.
- `human_reference_reply` — what we'd actually send, written independently
  of any model output (labeled BEFORE running the agent on these examples,
  to avoid anchoring).

## Two-pass workflow for judge/human agreement

1. **Pass 1** (before touching the agent): label `true_intent`,
   `true_action`, `human_reference_reply` for all 200 rows. This produces
   `golden_eval_labeled.csv` with `human_judge_score` left blank.
2. Run the full pipeline (`src/eval/run_eval.py`), which drafts replies and
   runs the LLM judge on a sample.
3. **Pass 2**: re-open `scripts/label_golden_set.py` (or edit the CSV
   directly) and hand-score a subsample of the *agent's actual draft
   replies* 1-5, filling in `human_judge_score`, independently of what the
   LLM judge said. Re-run `src/eval/run_eval.py` — it will now also print
   the judge/human agreement section.

This ordering matters: labeling reference replies and true intents
*before* seeing agent output avoids anchoring bias in the "true" labels;
scoring agent replies *after* the LLM judge already scored them (but
without looking at the judge's score) avoids anchoring bias in the
agreement check.

## Known limitations of this golden set

- Single labeler, no inter-annotator agreement computed (would need a
  second labeler — see report/REPORT.md, "what we'd do next").
- 200 examples means per-class counts for the rarer intents
  (`account_access`, `billing_inquiry`) are small (~15-20 each), so
  per-class F1 for those has wide uncertainty — flagged explicitly in the
  report rather than glossed over.
