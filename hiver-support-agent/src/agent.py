"""
The support agent itself. Orchestrates:

  1. classify intent  (src/intents.py)
  2. retrieve similar historically-resolved threads (src/retrieval.py)
  3. draft a reply grounded in those retrieved examples (this file)
  4. decide auto-handle vs escalate, WITH a stated reason (this file)

This is deliberately a plain Python class, not a framework (no LangChain /
agent-framework abstraction) -- for a task this shaped (fixed 3-step
pipeline, not open-ended tool use), a framework adds indirection without
adding capability. See decision_log.md, decision #4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src import config
from src.intents import LLMIntentClassifier
from src.llm_client import chat
from src.retrieval import ThreadRetriever

log = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    customer_message: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    retrieved_examples: list[dict]
    draft_reply: str
    action: str            # "auto_handle" | "escalate"
    escalation_reason: str  # empty string if auto_handle
    grounding_score: float = 0.0  # avg similarity of retrieved examples, 0 if none found


_DRAFT_SYSTEM_PROMPT = f"""You are drafting a customer support reply on behalf of \
{config.BRAND}, replying on Twitter (280 char public reply norms: be concise, \
professional, empathetic, and if the issue needs private info, ask the customer to move \
to DM). You are shown real examples of how {config.BRAND} has resolved similar issues in \
the past -- match that tone and those resolution patterns. Do not invent policies, refund \
amounts, or timelines that are not supported by the examples or standard customer-support \
practice. If you are not confident a good grounded reply is possible, say so honestly \
rather than fabricating specifics."""

_DRAFT_USER_TEMPLATE = """Customer's intent (classified): {intent}
Customer message: {message}

Examples of how {brand} has resolved similar issues before:
{examples}

Write a short reply {brand} could send. Return JSON:
{{"reply": "<the reply text>", "confident_in_grounding": <true/false>, \
"note_if_unconfident": "<why, or empty string>"}}"""


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(no closely similar historical examples found)"
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"{i}. Customer: {ex['customer_message']}\n   {config.BRAND} replied: "
            f"{ex['brand_reply']} (similarity={ex['similarity']:.2f})"
        )
    return "\n".join(blocks)


class SupportAgent:
    def __init__(self, retriever: ThreadRetriever, provider: str | None = None):
        self.retriever = retriever
        self.provider = provider or config.LLM_PROVIDER
        self.classifier = LLMIntentClassifier(provider=self.provider)

    def _draft_reply(self, message: str, intent: str, examples: list[dict]) -> tuple[str, bool, str]:
        from src.llm_client import chat_json
        user = _DRAFT_USER_TEMPLATE.format(
            intent=intent, message=message, brand=config.BRAND,
            examples=_format_examples(examples),
        )
        try:
            result = chat_json(_DRAFT_SYSTEM_PROMPT, user, provider=self.provider, temperature=0.3)
            return (
                result.get("reply", "").strip(),
                bool(result.get("confident_in_grounding", False)),
                result.get("note_if_unconfident", ""),
            )
        except Exception as e:  # noqa: BLE001
            log.error("Reply drafting failed: %s", e)
            return "", False, f"drafting error: {e}"

    def _decide_action(
        self, intent: str, intent_confidence: float, grounding_score: float,
        model_confident_in_grounding: bool, drafting_note: str,
    ) -> tuple[str, str]:
        """Escalation policy. Every branch returns a human-readable reason --
        the assignment explicitly requires a *stated* reason, not just a
        label. See decision_log.md, decisions #6-#8 for why these specific
        thresholds/rules were chosen over a learned escalation classifier."""
        if intent in config.ALWAYS_ESCALATE_INTENTS:
            return "escalate", f"Intent '{intent}' is policy-flagged for mandatory human review."
        if intent_confidence < config.INTENT_CONFIDENCE_ESCALATION_THRESHOLD:
            return "escalate", (
                f"Intent classification confidence ({intent_confidence:.2f}) is below the "
                f"{config.INTENT_CONFIDENCE_ESCALATION_THRESHOLD} threshold -- message is "
                "ambiguous or multi-issue."
            )
        if grounding_score < 0.35:
            return "escalate", (
                f"No closely similar historical resolution found (top similarity "
                f"{grounding_score:.2f}) -- reply would not be well-grounded."
            )
        if not model_confident_in_grounding:
            return "escalate", f"Drafting model flagged low confidence: {drafting_note or 'unspecified'}"
        return "auto_handle", ""

    def handle(self, message: str, prior_context: str = "") -> AgentResponse:
        intent_result = self.classifier.classify(message, prior_context)
        examples = self.retriever.query(message)
        grounding_score = sum(e["similarity"] for e in examples) / len(examples) if examples else 0.0

        reply, model_confident, note = self._draft_reply(message, intent_result["intent"], examples)

        action, reason = self._decide_action(
            intent_result["intent"], intent_result["confidence"], grounding_score,
            model_confident, note,
        )

        return AgentResponse(
            customer_message=message,
            intent=intent_result["intent"],
            intent_confidence=intent_result["confidence"],
            intent_reasoning=intent_result["reasoning"],
            retrieved_examples=examples,
            draft_reply=reply,
            action=action,
            escalation_reason=reason,
            grounding_score=grounding_score,
        )
