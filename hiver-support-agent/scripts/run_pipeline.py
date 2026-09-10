"""
Convenience end-to-end runner. Runs, in order:

  1. data_prep       (raw twcs.csv -> brand thread pairs)
  2. retrieval index build
  3. a quick demo of the agent on 3 hand-picked example messages

This is NOT the evaluation harness (that's src/eval/run_eval.py, which
needs the golden set). This script exists so a reviewer can see the agent
actually produce output within minutes of cloning, before investing time
in hand-labeling anything.

Run: python scripts/run_pipeline.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_prep, retrieval  # noqa: E402
from src.agent import SupportAgent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEMO_MESSAGES = [
    ("My order was supposed to arrive 3 days ago and tracking hasn't updated at all. "
     "This is the second time this month, I need a real answer.", ""),
    ("hey do you guys ship to canada?", ""),
    ("I've emailed twice and called once about my damaged package and nobody has helped me. "
     "I want a refund NOW or I'm disputing the charge with my bank.", ""),
]


def main():
    if not config.BRAND_THREADS_PARQUET.exists():
        log.info("Processed threads not found, running data_prep...")
        data_prep.main()

    if not (Path(str(config.RETRIEVAL_INDEX_DIR) + "_emb.npy")).exists():
        log.info("Retrieval index not found, building it...")
        retrieval.build_and_save_index()

    retriever = retrieval.ThreadRetriever.load(config.RETRIEVAL_INDEX_DIR)
    agent = SupportAgent(retriever)

    for message, ctx in DEMO_MESSAGES:
        print("\n" + "=" * 70)
        print(f"CUSTOMER: {message}")
        result = agent.handle(message, ctx)
        print(f"  intent:     {result.intent}  (confidence={result.intent_confidence:.2f})")
        print(f"  action:     {result.action}")
        if result.action == "escalate":
            print(f"  reason:     {result.escalation_reason}")
        print(f"  grounding:  {result.grounding_score:.2f} "
              f"({len(result.retrieved_examples)} similar past threads found)")
        print(f"  draft reply:\n    {result.draft_reply}")


if __name__ == "__main__":
    main()
