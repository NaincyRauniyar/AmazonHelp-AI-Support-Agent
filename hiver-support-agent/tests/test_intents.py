"""
Unit tests that don't require an LLM call or the real dataset -- these
should pass with just `pip install -r requirements.txt`, no API keys, no
Ollama running. LLM-dependent behavior is covered separately via the demo
script (scripts/run_pipeline.py), which is inherently an integration test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intents import TfidfIntentClassifier  # noqa: E402


def test_tfidf_classifier_fits_and_predicts():
    messages = [
        "my package is late and tracking hasn't updated",
        "where is my order, it's been a week",
        "I need a refund for my order",
        "please refund my money, wrong charge",
        "can't log into my account, password reset broken",
        "my account is locked and I can't access it",
    ]
    labels = [
        "delivery_delay", "delivery_delay",
        "refund_request", "refund_request",
        "account_access", "account_access",
    ]
    clf = TfidfIntentClassifier().fit(messages, labels)
    result = clf.classify("my order still hasn't arrived and tracking is stuck")
    assert result["intent"] in {"delivery_delay", "refund_request", "account_access"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_tfidf_classifier_raises_if_not_fitted():
    clf = TfidfIntentClassifier()
    try:
        clf.classify("test message")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


if __name__ == "__main__":
    test_tfidf_classifier_fits_and_predicts()
    test_tfidf_classifier_raises_if_not_fitted()
    print("All intent tests passed.")
