# Report — AmazonHelp AI Support Agent

> **Fill-in-the-numbers template.** Every `[FILL IN]` marker below gets
> replaced with real output from `python -m src.eval.run_eval` once you've
> run it against the real Kaggle dataset + your chosen free LLM provider.
> Do not submit with markers still in it — see README > "Get to a
> submittable report" for the exact command sequence. Sections without a
> `[FILL IN]` marker are already complete and don't need editing (though
> you're free to add real observations as you actually see the data).

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp's Twitter support handles
a high volume of short, often impatient public complaints across a wide
spread of issue types (delivery, refunds, defects, account access,
billing). For this brand, "good" means:

- **High escalation recall on the issues that actually need a human**
  (account security, genuinely angry/repeat-contact customers, anything
  the agent isn't well-grounded on) — a wrong auto-reply on a bad issue is
  reputationally worse than a slower human reply.
- **A reasonable auto-handle rate** on the large volume of routine,
  well-precedented issues (where's my order, how do I return this) — the
  whole point of the system is to not send every one of these to a human.
- **Grounded, not hallucinated, replies** — no invented refund amounts,
  policies, or timelines. This matters more than reply eloquence.

**What we chose not to build:**
- No multi-turn conversation state / follow-up handling — the agent
  classifies and drafts for a single customer message (with one turn of
  prior context if available), not an ongoing back-and-forth. A real
  deployment would need conversation memory; out of scope for a take-home.
- No actual sending / Twitter API integration — this produces a draft and
  a routing decision, which is what the assignment asks for ("draft a
  reply", "decide whether... auto-handled or escalated"), not a live bot.
- No fine-tuning — zero-shot LLM classification + retrieval-grounded
  generation, both of which work without any labeled training data on day
  one. Fine-tuning would need far more labeled data than a 150-250 example
  golden set provides, and would defeat the "works from day one on a new
  brand" property that makes this approach viable.
- No multi-brand generalization work — the whole pipeline is brand-scoped
  by design (config.BRAND), since resolution patterns genuinely differ
  brand to brand; that's a feature, not a shortcut.

## 2. Results vs. baselines

Run: `python -m src.eval.run_eval --judge-n 40`

### Intent classification

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Agent (LLM zero-shot) | [FILL IN] | [FILL IN] | [FILL IN] |
| Simple baseline (TF-IDF + LogReg) | [FILL IN] | [FILL IN] | [FILL IN] |
| Trivial baseline (majority class) | [FILL IN] | [FILL IN] | [FILL IN] |

[FILL IN: 2-3 sentences on what this shows — e.g. does the zero-shot
classifier actually beat a baseline trained directly on the golden set? If
not, that's a real, reportable finding, not something to hide.]

### Escalation decision

Positive class = `escalate`.

| Metric | Value |
|---|---|
| Precision | [FILL IN] |
| Recall | [FILL IN] |
| F1 | [FILL IN] |
| Predicted auto-handle rate | [FILL IN] |
| True auto-handle rate (per golden labels) | [FILL IN] |

[FILL IN: is the gap between predicted and true auto-handle rate small? A
large gap either direction means the threshold in `HIVER_CONF_THRESHOLD`
is miscalibrated for this data — say so plainly if it is.]

### Reply quality (LLM-as-judge, 1-5 scale, n=[FILL IN])

| Method | Relevance | Groundedness | Tone | Actionability | Overall |
|---|---|---|---|---|---|
| Agent (retrieval + generation) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Retrieval-only baseline (nearest historical reply, no generation) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Trivial baseline (most common historical reply) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

### Judge / human agreement

From `src/eval/metrics.judge_human_agreement`, computed on the hand-scored
subsample (see `data/golden_eval/labeling_guide.md`, "two-pass workflow").

| Metric | Value |
|---|---|
| Pearson r | [FILL IN] |
| Spearman r | [FILL IN] |
| Exact agreement rate | [FILL IN] |
| Within-1-point agreement rate | [FILL IN] |
| n | [FILL IN] |

[FILL IN: if within-1-point agreement is below ~0.7, the judge scores in
this report should be read as directional, not precise — say so.]

## 3. Failure analysis — top 5 failure modes

[FILL IN once you've read through `data/processed/eval_results.csv` and
`data/processed/judge_scores.csv` after a real run. For each mode: name
it, give 1-2 real (message, agent output) examples pulled directly from
those CSVs, and a hypothesis for why it happens. Template below —
replace every bracket with what you actually observe; do not invent
examples.]

### Failure mode 1: [name]
- **Example:** customer said "[real message]" — agent [intent/reply/
  escalation output], which was wrong because [reason].
- **Hypothesis:** [why this happens — e.g. retrieval corpus has no similar
  historical example because this issue type is rare in the subsample;
  intent boundary is genuinely ambiguous; classifier prompt lacks an
  example of this phrasing pattern; etc.]

### Failure mode 2: [name]
...

### Failure mode 3: [name]
...

### Failure mode 4: [name]
...

### Failure mode 5: [name]
...

## 4. What is misleading about my headline number?

*(Mandatory section — answer honestly, don't just restate limitations
generically.)*

Known, structural reasons the headline accuracy/F1/judge-overall numbers
above overstate real-world performance:

- **The TF-IDF baseline is trained and evaluated on overlapping data
  distribution** (both drawn from the same 200-example golden set via a
  fit-then-predict split that [FILL IN: state whether you did a proper
  train/test split within the golden set, or evaluated in-sample — if
  in-sample, this baseline's number is optimistic and should be labeled as
  such]).
- **Single labeler, no inter-annotator agreement** — the "true" intent and
  escalation labels reflect one person's judgment calls on genuinely
  ambiguous messages. A second labeler would likely disagree on some
  fraction of the harder examples (the long/multi-issue stratum
  especially), which would lower measured accuracy for both the agent and
  every baseline.
- **The escalation confidence threshold (0.55) was picked before seeing
  results and left untuned** (decision_log.md #6) — this is honest but
  means the reported escalation precision/recall is not the best this
  approach could do with minimal tuning, nor is it validated on a held-out
  split.
- **The dataset is subsampled** ([FILL IN: your MAX_THREADS value] out of
  the brand's full historical volume) — rare intents and unusual phrasings
  are underrepresented in both the retrieval corpus and the golden set
  relative to true production volume.
- **LLM judge and agent [FILL IN: do / do not] use the same underlying
  model** — if the same, judge scores likely have some self-grading
  generosity bias (decision_log.md #11).
- [FILL IN: any additional issue you actually notice in your run —
  e.g. the judge consistently over-scoring short generic replies, a
  particular intent the classifier systematically confuses, etc.]

## 5. What we'd do next with one more week

1. Second labeler + inter-annotator agreement (Cohen's kappa) on the
   golden set, to know how much of the "error" is actually label noise.
2. Proper train/held-out split for the TF-IDF baseline and for tuning
   `HIVER_CONF_THRESHOLD`, instead of the single-pass evaluation used here.
3. Expand the retrieval corpus beyond the subsample — measure whether
   grounding score / reply quality improves with more historical threads
   indexed, and find the point of diminishing returns.
4. Add conversation-level context (not just one prior turn) for the
   genuinely multi-turn threads in the dataset, and re-measure whether that
   changes intent accuracy on the "long message" golden-set stratum
   specifically.
5. A/B a cross-encoder reranker on top of the current bi-encoder retrieval
   (sentence-transformers `all-MiniLM-L6-v2`) to see if grounding-score
   quality (not just speed) improves enough to matter.
6. Try a second, larger free-tier model as the drafting LLM and compare
   judge scores — current default is a small/fast model chosen for
   free-tier throughput, not necessarily the best available for free.
