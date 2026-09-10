import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import ThreadRetriever  # noqa: E402


def _sample_threads():
    return pd.DataFrame([
        {"customer_message": "my package never arrived and tracking is stuck", "brand_reply": "Sorry about that, checking now.", "thread_id": 1},
        {"customer_message": "I want a refund, wrong item sent", "brand_reply": "We'll process a refund right away.", "thread_id": 2},
        {"customer_message": "can't log into my account", "brand_reply": "Please DM us to verify your identity.", "thread_id": 3},
    ])


def test_retriever_returns_most_similar_first():
    retriever = ThreadRetriever(_sample_threads()).build()
    results = retriever.query("where is my order, it hasn't shipped", top_k=2)
    assert len(results) == 2
    # the delivery-related historical message should rank first
    assert "package" in results[0]["customer_message"] or "arrived" in results[0]["customer_message"]
    assert results[0]["similarity"] >= results[1]["similarity"]


if __name__ == "__main__":
    test_retriever_returns_most_similar_first()
    print("All retrieval tests passed.")
