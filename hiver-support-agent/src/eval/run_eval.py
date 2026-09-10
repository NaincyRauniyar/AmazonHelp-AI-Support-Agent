"""
The evaluation harness. Runs:

  - the real agent
  - trivial baseline (intent + reply)
  - simple baseline (intent + reply)

against the hand-labeled golden set, computes automated metrics for all
three, and runs the LLM judge on a sample of replies (judge calls cost
money/time even on free tiers with rate limits, so we don't judge every
row by default -- see --judge-n).

Run: python -m src.eval.run_eval --judge-n 40
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src import config
from src.agent import SupportAgent
from src.baselines import RetrievalOnlyReplyBaseline, TrivialIntentBaseline, TrivialReplyBaseline
from src.eval.llm_judge import judge_reply
from src.eval.metrics import escalation_metrics, intent_metrics, judge_human_agreement
from src.intents import TfidfIntentClassifier
from src.retrieval import ThreadRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_golden() -> pd.DataFrame:
    if not config.GOLDEN_LABELED_CSV.exists():
        raise FileNotFoundError(
            f"{config.GOLDEN_LABELED_CSV} not found. Run scripts/label_golden_set.py "
            "first (see README > 'Build the golden set')."
        )
    df = pd.read_csv(config.GOLDEN_LABELED_CSV)
    required = {"customer_message", "prior_context", "true_intent", "true_action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Golden set is missing required columns: {missing}")
    return df


def run(judge_n: int, provider: str | None):
    golden = load_golden()
    threads = pd.read_parquet(config.BRAND_THREADS_PARQUET)
    retriever = ThreadRetriever.load(config.RETRIEVAL_INDEX_DIR)

    log.info("Running agent on %d golden examples...", len(golden))
    agent = SupportAgent(retriever, provider=provider)
    tfidf = TfidfIntentClassifier().fit(golden["customer_message"].tolist(), golden["true_intent"].tolist())
    trivial_intent = TrivialIntentBaseline().fit(golden["true_intent"].tolist())
    trivial_reply = TrivialReplyBaseline(threads)
    retrieval_only_reply = RetrievalOnlyReplyBaseline(retriever)

    rows = []
    for _, row in golden.iterrows():
        msg, ctx = row["customer_message"], row.get("prior_context", "")
        agent_out = agent.handle(msg, ctx)
        tfidf_intent = tfidf.classify(msg, ctx)
        rows.append({
            "customer_message": msg,
            "true_intent": row["true_intent"],
            "true_action": row["true_action"],
            "agent_intent": agent_out.intent,
            "agent_action": agent_out.action,
            "agent_escalation_reason": agent_out.escalation_reason,
            "agent_reply": agent_out.draft_reply,
            "agent_grounding_score": agent_out.grounding_score,
            "tfidf_intent": tfidf_intent["intent"],
            "trivial_intent": trivial_intent.classify(msg)["intent"],
            "trivial_reply": trivial_reply.reply(msg),
            "retrieval_only_reply": retrieval_only_reply.reply(msg),
            "human_reference_reply": row.get("human_reference_reply", ""),
        })
    results = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.EVAL_RESULTS_CSV, index=False)
    log.info("Saved raw eval results -> %s", config.EVAL_RESULTS_CSV)

    print("\n" + "=" * 70)
    print("INTENT CLASSIFICATION")
    print("=" * 70)
    for name, col in [("Agent (LLM zero-shot)", "agent_intent"),
                       ("Simple baseline (TF-IDF+LogReg)", "tfidf_intent"),
                       ("Trivial baseline (majority class)", "trivial_intent")]:
        m = intent_metrics(results["true_intent"], results[col])
        print(f"\n-- {name} --")
        print(f"  accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  weighted_f1={m['weighted_f1']:.3f}")

    print("\n" + "=" * 70)
    print("ESCALATION DECISION  (positive class = 'escalate')")
    print("=" * 70)
    m = escalation_metrics(results["true_action"], results["agent_action"])
    for k, v in m.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 70)
    print(f"REPLY QUALITY (LLM-as-judge, sampled n={judge_n})")
    print("=" * 70)
    sample = results.sample(min(judge_n, len(results)), random_state=config.RANDOM_SEED)
    judged = []
    for _, r in sample.iterrows():
        j = judge_reply(r["customer_message"], r["agent_reply"], r["human_reference_reply"], provider=provider)
        j["customer_message"] = r["customer_message"]
        j["reply"] = r["agent_reply"]
        judged.append(j)
    judged_df = pd.DataFrame(judged)
    judge_out_path = config.PROCESSED_DIR / "judge_scores.csv"
    judged_df.to_csv(judge_out_path, index=False)
    print(judged_df[["relevance", "groundedness", "tone", "actionability", "overall"]].mean().round(2))
    log.info("Saved judge scores -> %s", judge_out_path)

    if "human_judge_score" in golden.columns:
        print("\n" + "=" * 70)
        print("JUDGE <-> HUMAN AGREEMENT")
        print("=" * 70)
        merged = sample.merge(
            golden[["customer_message", "human_judge_score"]], on="customer_message", how="left"
        )
        agreement = judge_human_agreement(judged_df["overall"].tolist(), merged["human_judge_score"].tolist())
        for k, v in agreement.items():
            print(f"  {k}: {v}")
    else:
        print(
            "\n[!] No 'human_judge_score' column in golden set -- judge/human agreement not "
            "computed. Hand-score a subset of judged replies 1-5 yourself and add that column "
            "to re-run this section (see README > 'Judge/human agreement')."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-n", type=int, default=40, help="How many replies to send to the LLM judge")
    parser.add_argument("--provider", type=str, default=None, help="Override LLM provider for this run")
    args = parser.parse_args()
    run(args.judge_n, args.provider)
