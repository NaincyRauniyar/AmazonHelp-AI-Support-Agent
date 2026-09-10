"""
Intent classification.

Two implementations, used at different points in the assignment:

1. `LLMIntentClassifier`   -- zero-shot, LLM-prompted. This is the agent's
   real classifier: no labeled training data required, works on day one.
2. `TfidfIntentClassifier` -- the "simple baseline" required by the report.
   Trained on the golden set itself (small-n, but that's what a real team
   would have on day one too) using TF-IDF + Logistic Regression.

A third, even dumber "trivial baseline" (always predict the majority
class) lives in src/baselines.py next to the trivial reply baseline, since
the report wants both baselines compared in one place.
"""
from __future__ import annotations

import logging

from src import config
from src.llm_client import chat_json

log = logging.getLogger(__name__)

_INTENT_LIST_STR = "\n".join(f"- {i}" for i in config.INTENTS)

_SYSTEM_PROMPT = f"""You are an intent classifier for {config.BRAND} customer support \
messages sent over Twitter. Classify the customer's message into exactly one of the \
following intents:

{_INTENT_LIST_STR}

Rules:
- Pick the SINGLE best-fitting intent, even if the message could arguably fit two.
- "complaint_escalation" is for messages that are primarily angry/frustrated, mention \
  repeated unresolved contact, or explicitly ask for a manager/escalation -- not just any \
  negative message.
- If nothing fits well, use "general_inquiry".
- Return your confidence (0.0-1.0) in your own classification honestly. Low confidence is \
  fine and expected for ambiguous or multi-issue messages.
"""

_USER_TEMPLATE = """Prior context (may be empty): {prior_context}
Customer message: {message}

Return JSON: {{"intent": "<one of the intents>", "confidence": <float 0-1>, "reasoning": "<one short sentence>"}}"""


class LLMIntentClassifier:
    def __init__(self, provider: str | None = None):
        self.provider = provider or config.LLM_PROVIDER

    def classify(self, message: str, prior_context: str = "") -> dict:
        user = _USER_TEMPLATE.format(prior_context=prior_context or "(none)", message=message)
        try:
            result = chat_json(_SYSTEM_PROMPT, user, provider=self.provider)
        except Exception as e:  # noqa: BLE001 - degrade gracefully, never crash the agent
            log.error("Intent classification failed, defaulting to general_inquiry: %s", e)
            return {"intent": "general_inquiry", "confidence": 0.0, "reasoning": f"classifier error: {e}"}

        intent = result.get("intent", "general_inquiry")
        if intent not in config.INTENTS:
            log.warning("Model returned unknown intent '%s', coercing to general_inquiry", intent)
            intent = "general_inquiry"
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return {"intent": intent, "confidence": confidence, "reasoning": result.get("reasoning", "")}


class TfidfIntentClassifier:
    """Simple, cheap, fully local baseline trained on hand-labeled data.
    Required by the report as the "simple baseline" (vs. the LLM zero-shot
    classifier, and vs. the trivial majority-class baseline)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def fit(self, messages: list[str], labels: list[str]):
        X = self.vectorizer.fit_transform(messages)
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def classify(self, message: str, prior_context: str = "") -> dict:
        if not self._fitted:
            raise RuntimeError("TfidfIntentClassifier must be .fit() before use")
        X = self.vectorizer.transform([message])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X).max()
        return {"intent": pred, "confidence": float(proba), "reasoning": "tfidf+logreg baseline"}
