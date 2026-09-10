"""
LLM-as-judge for reply quality. Scores a drafted reply against 4 rubric
dimensions (1-5 each), plus an overall pass/fail, given the customer
message and (if available) the historical ground-truth resolution as a
reference point.

Rubric dimensions (kept to 4, deliberately small -- see decision_log.md
decision #10 for why we didn't do 8+ dimensions):

  - relevance:    does the reply actually address what the customer asked?
  - groundedness: does it stick to real, supportable policy/info, no
                   invented specifics (refund amounts, timelines, etc.)?
  - tone:         professional, empathetic, on-brand for a support reply?
  - actionability: does the customer know what happens next / what to do?

We ask for a short justification per dimension too -- ungrounded numeric
scores from an LLM judge are close to useless for failure analysis.
"""
from __future__ import annotations

from src import config
from src.llm_client import chat_json

_SYSTEM_PROMPT = f"""You are an expert QA reviewer grading a draft customer-support reply \
for {config.BRAND}. Score strictly -- a reply that is merely "fine" should NOT get top \
marks. Be skeptical of confident-sounding but unsupported claims (invented refund amounts, \
made-up timelines, policies not evidenced anywhere)."""

_USER_TEMPLATE = """Customer message: {message}

Draft reply being graded: {reply}

Reference: how this brand has resolved similar issues before (for grounding check only, \
not a required exact match): {reference}

Score each dimension 1 (poor) to 5 (excellent):
- relevance: does the reply address the customer's actual issue?
- groundedness: is everything in the reply supportable by the reference/standard practice, \
with no invented specifics?
- tone: professional and empathetic?
- actionability: is it clear what happens next?

Return JSON:
{{"relevance": <int>, "groundedness": <int>, "tone": <int>, "actionability": <int>, \
"overall": <int 1-5, your holistic judgment, not just the average>, \
"justification": "<1-2 sentences>"}}"""


def judge_reply(message: str, reply: str, reference: str = "", provider: str | None = None) -> dict:
    provider = provider or config.JUDGE_LLM_PROVIDER
    user = _USER_TEMPLATE.format(
        message=message, reply=reply or "(empty -- drafting failed)",
        reference=reference or "(none available)",
    )
    try:
        result = chat_json(_SYSTEM_PROMPT, user, provider=provider, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return {
            "relevance": 0, "groundedness": 0, "tone": 0, "actionability": 0, "overall": 0,
            "justification": f"judge error: {e}",
        }
    for key in ("relevance", "groundedness", "tone", "actionability", "overall"):
        result[key] = int(result.get(key, 0))
    return result
