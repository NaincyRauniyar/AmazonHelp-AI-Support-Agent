"""
The report requires results against "at least two baselines (a trivial one
and a simple one)". We define both, for BOTH sub-tasks (intent + reply),
since a real reviewer will want to know the agent beats naive alternatives
on every axis it's graded on, not just intent accuracy.

Trivial baselines:
  - intent:  always predict the majority class from the golden set.
  - reply:   always send the single most common canned reply in the brand's
             history (mode of brand_reply, near-dupes collapsed).

Simple baselines:
  - intent:  TF-IDF + Logistic Regression (src/intents.TfidfIntentClassifier).
  - reply:   nearest-neighbor retrieval with NO generation -- just return
             the single most similar historical brand_reply verbatim
             (retrieval without generation).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from src.retrieval import ThreadRetriever


class TrivialIntentBaseline:
    def __init__(self):
        self.majority_label = None

    def fit(self, labels: list[str]):
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def classify(self, message: str, prior_context: str = "") -> dict:
        return {"intent": self.majority_label, "confidence": 1.0, "reasoning": "majority class"}


class TrivialReplyBaseline:
    """Always the single most frequent brand reply in the historical data."""

    def __init__(self, threads: pd.DataFrame):
        self.canned_reply = threads["brand_reply"].mode().iloc[0]

    def reply(self, message: str) -> str:
        return self.canned_reply


class RetrievalOnlyReplyBaseline:
    """Nearest-neighbor retrieval, no LLM generation: return the historical
    brand reply to the single most similar past customer message, verbatim."""

    def __init__(self, retriever: ThreadRetriever):
        self.retriever = retriever

    def reply(self, message: str) -> str:
        hits = self.retriever.query(message, top_k=1)
        if not hits:
            return ""
        return hits[0]["brand_reply"]
