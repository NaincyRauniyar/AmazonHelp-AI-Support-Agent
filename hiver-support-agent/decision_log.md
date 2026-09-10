# Decision Log

Non-obvious calls made while building this, and why. Roughly in the order
they came up.

1. **Brand: AmazonHelp.** Picked over smaller brands in the dataset because
   it has enough volume to build a real retrieval corpus (thousands of
   resolved pairs) even after aggressive subsampling, and its issues span
   nearly all 9 intents we defined, rather than being dominated by one
   category (like a single-product brand might be).

2. **9 intents, defined by reading real data, not by guessing.** Read
   ~150 raw AmazonHelp threads before writing `src/config.py`. Rejected an
   early 15-intent draft — too many categories had near-zero examples and
   the boundary between them was unclear even to a human, which meant the
   LLM classifier would be unclear too. Fewer, cleaner categories beat
   more, blurrier ones.

3. **No agent framework (LangChain/etc).** The pipeline is a fixed 3-step
   sequence (classify -> retrieve -> draft), not open-ended tool use. A
   framework adds indirection, dependency weight, and a debugging layer
   without adding capability here. Plain Python classes are easier to
   reason about and to explain live.

4. **Escalation is a rule-based policy over model outputs, not a 4th LLM
   call.** Considered asking the LLM "should this be escalated?" directly,
   but that just relocates the judgment into another opaque model call
   with no auditable reason. Instead, escalation is 4 explicit, orderable
   rules (see `src/agent.py::_decide_action`) over signals we already
   compute: intent, intent confidence, retrieval grounding score, and the
   drafting model's own stated confidence. Every escalation has a reason
   that traces to one specific signal — required by the assignment, and
   also just easier to debug and tune later.

5. **Golden set stratified by message length, not by intent.** Can't
   stratify by ground-truth intent before you have ground-truth intents —
   that's circular. Length is a free, computable-in-advance proxy that
   correlates with case complexity (short = simple ask, long = messy
   multi-issue). See `data/golden_eval/labeling_guide.md` for the full
   sampling writeup.

6. **Escalation confidence threshold set at 0.55, not tuned to hit a
   specific auto-handle rate.** Picked as a reasonable-looking midpoint
   before seeing golden-set results, then deliberately left unturned in
   the reference run so the reported escalation-recall number isn't
   circularly optimized against the same data it's evaluated on. The
   report's "what's misleading" section flags this as a knob worth tuning
   with a proper held-out validation split if this went further.

7. **Two intents (`complaint_escalation`, `account_access`) are always
   escalated, regardless of confidence.** Not a coverage-maximizing
   choice — it costs auto-handle rate. Justification: an angry/repeat
   customer being auto-replied to with a canned-sounding grounded reply is
   a worse outcome than a slightly slower human response, and account
   security issues (locked accounts, suspected unauthorized access) are a
   correctness/liability issue, not a tone issue — never appropriate to
   resolve via a public Twitter reply either.

8. **Grounding score = average retrieval similarity of top-k, with a hard
   floor (0.35) below which the agent always escalates**, rather than
   trusting the drafting LLM's self-reported confidence alone. LLMs are
   fairly bad at knowing when they're about to hallucinate a policy detail;
   a similarity floor on the retrieved evidence is a cheaper, more
   reliable signal for "do we actually have grounding for this."

9. **Retrieval is flat numpy cosine similarity, not a vector DB.** At the
   scale of a subsampled single-brand corpus (thousands, not millions, of
   rows), a vector DB (FAISS/Chroma/Pinecone) is unneeded infrastructure.
   Revisit if a brand's full thread history needs indexing without
   subsampling.

10. **LLM-judge rubric has 4 dimensions, not 8+.** Tried an earlier 8-
    dimension rubric (relevance, groundedness, tone, actionability,
    completeness, conciseness, policy-compliance, brand-voice) and found
    the judge's scores across the extra dimensions were highly correlated
    with each other (i.e., not adding independent signal) while taking
    ~2x the judge calls. Collapsed to the 4 that showed the most
    independent variance in a pilot run of 20 examples.

11. **Judge and agent can use different LLM providers** (`HIVER_JUDGE_
    PROVIDER` separate from `HIVER_LLM_PROVIDER`), to reduce self-grading
    bias where a model rates its own outputs generously. Left the same by
    default only because requiring two separate free-tier signups is
    friction the assignment doesn't ask for — flagged explicitly as a
    "what's misleading" caveat in the report.

12. **Trivial reply baseline is "most frequent historical reply," not
    "no reply" or a fixed generic string.** A truly trivial baseline
    should still be something a lazy-but-real system might do (send the
    single most common canned response ever sent), not a strawman. Makes
    the comparison to the real agent more honest.

13. **Bundled a small (30-example) hand-written sample dataset
    (`scripts/make_sample_data.py`)** so the repo is runnable within the
    README's 15-minute reproduction window without a Kaggle account or a
    2.6GB download. Explicitly labeled as a demo aid, not a substitute for
    the real data the golden set and reported numbers are built from — see
    README > "Two ways to run this."

14. **TF-IDF+LogReg simple baseline is trained on the golden set itself**
    (small-n, ~150-250 examples), not on a separately-labeled larger
    training set. This is intentionally realistic: a team standing this up
    in week one has exactly the labeled data they hand-built for eval and
    nothing else. Report flags the resulting baseline variance honestly
    rather than hiding it behind a bigger, separately-collected training
    set that wouldn't reflect a real day-one deployment.
