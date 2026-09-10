"""
Interactive CLI for building the 150-250 example hand-labeled golden set.

Sampling strategy (see README > 'Build the golden set' and decision_log.md
decision #5 for the reasoning):
  - Stratified by a cheap keyword-based pre-tag, so the golden set isn't
    dominated by whatever the most common issue type happens to be.
  - Includes a deliberate slice of "hard" examples: very short messages,
    very long/multi-issue messages, and messages with no close historical
    neighbor -- because those are exactly where the agent is most likely
    to fail, and a golden set that's 90% easy cases won't catch it.

Run: python scripts/label_golden_set.py --n 200
Resumes automatically if run again (skips already-labeled rows).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

INTENT_MENU = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(config.INTENTS))


def sample_for_labeling(threads: pd.DataFrame, n: int) -> pd.DataFrame:
    df = threads.copy()
    df["msg_len"] = df["customer_message"].str.len()

    # crude strata: short / medium / long messages, so the golden set has
    # a mix of trivial one-liners and messy multi-issue rambles
    short = df[df["msg_len"] < 60]
    medium = df[(df["msg_len"] >= 60) & (df["msg_len"] < 150)]
    long_ = df[df["msg_len"] >= 150]

    n_each = n // 3
    parts = [
        short.sample(min(n_each, len(short)), random_state=config.RANDOM_SEED),
        medium.sample(min(n_each, len(medium)), random_state=config.RANDOM_SEED),
        long_.sample(min(n - 2 * n_each, len(long_)), random_state=config.RANDOM_SEED),
    ]
    sampled = pd.concat(parts).drop_duplicates(subset=["customer_message"])
    return sampled.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)  # shuffle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    threads = pd.read_parquet(config.BRAND_THREADS_PARQUET)

    if config.GOLDEN_LABELED_CSV.exists():
        labeled = pd.read_csv(config.GOLDEN_LABELED_CSV)
        print(f"Resuming: {len(labeled)} examples already labeled.")
    else:
        to_label = sample_for_labeling(threads, args.n)
        labeled = pd.DataFrame(columns=[
            "customer_message", "prior_context", "true_intent", "true_action",
            "human_reference_reply", "human_judge_score", "notes",
        ])
        to_label.to_csv(config.GOLDEN_SET_CSV, index=False)
        print(f"Sampled {len(to_label)} examples -> {config.GOLDEN_SET_CSV}")

    pool = pd.read_csv(config.GOLDEN_SET_CSV)
    already = set(labeled["customer_message"]) if len(labeled) else set()
    remaining = pool[~pool["customer_message"].isin(already)]

    print(f"\n{len(remaining)} left to label. Ctrl+C anytime to save & quit.\n")
    print("Intents:\n" + INTENT_MENU)
    print("\nFor each example you'll enter: intent number, action (a=auto_handle/e=escalate),")
    print("a short reference reply a human agent would actually send, and optionally a")
    print("1-5 quality score you'd give the AGENT'S reply once you've run the pipeline once")
    print("(leave blank on first pass -- see README for the two-pass workflow).\n")

    new_rows = []
    try:
        for _, row in remaining.iterrows():
            print("-" * 70)
            if row["prior_context"] and isinstance(row["prior_context"], str):
                print(f"[prior] {row['prior_context']}")
            print(f"CUSTOMER: {row['customer_message']}")
            intent_in = input(f"Intent [1-{len(config.INTENTS)}]: ").strip()
            if not intent_in:
                continue
            intent = config.INTENTS[int(intent_in) - 1]
            action_in = input("Action [a=auto_handle / e=escalate]: ").strip().lower()
            action = "auto_handle" if action_in == "a" else "escalate"
            ref_reply = input("Reference reply a human would send: ").strip()
            notes = input("Notes (optional): ").strip()

            new_rows.append({
                "customer_message": row["customer_message"],
                "prior_context": row.get("prior_context", ""),
                "true_intent": intent,
                "true_action": action,
                "human_reference_reply": ref_reply,
                "human_judge_score": "",
                "notes": notes,
            })
    except KeyboardInterrupt:
        pass
    finally:
        if new_rows:
            out = pd.concat([labeled, pd.DataFrame(new_rows)], ignore_index=True)
            out.to_csv(config.GOLDEN_LABELED_CSV, index=False)
            print(f"\nSaved {len(out)} total labeled examples -> {config.GOLDEN_LABELED_CSV}")


if __name__ == "__main__":
    main()
