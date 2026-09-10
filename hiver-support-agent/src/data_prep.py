"""
Turn the raw Kaggle `twcs.csv` (Customer Support on Twitter) into a clean
table of (customer_message, brand_reply, thread_id) pairs for one brand.

The raw file is a flat table of individual tweets with `response_tweet_id`
and `in_response_to_tweet_id` columns that encode a reply graph. We walk
that graph to reconstruct customer -> brand reply pairs, which is the unit
everything downstream (retrieval, generation) operates on.

Run: python -m src.data_prep
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light normalization. We deliberately do NOT strip all punctuation /
    lowercase everything — LLMs work better with natural text, and we want
    the retrieval corpus to look like real tweets."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("[link]", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_raw() -> pd.DataFrame:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CSV} not found.\n"
            "Download the Kaggle 'Customer Support on Twitter' dataset "
            "(thoughtvector/customer-support-on-twitter), unzip it, and "
            "place twcs.csv at that path. See README > 'Get the data'."
        )
    log.info("Loading raw CSV (this file is ~2.6GB uncompressed, may take a bit)...")
    df = pd.read_csv(
        config.RAW_CSV,
        dtype={"tweet_id": "Int64", "in_response_to_tweet_id": "Int64",
               "response_tweet_id": str},
    )
    return df


def build_brand_threads(df: pd.DataFrame, brand: str, max_threads: int) -> pd.DataFrame:
    """Reconstruct (customer_msg -> brand_reply) pairs for one brand.

    A pair is: a non-inbound (brand) tweet whose `in_response_to_tweet_id`
    points at an inbound (customer) tweet. We also pull a short bit of
    preceding context (previous customer turn, if any) so the retrieval
    corpus captures multi-turn nuance, not just single messages.
    """
    df = df.copy()
    df["text"] = df["text"].map(clean_text)

    brand_replies = df[(df["author_id"] == brand) & df["in_response_to_tweet_id"].notna()]
    log.info("Found %d raw tweets from %s that are replies.", len(brand_replies), brand)

    tweets_by_id = df.set_index("tweet_id")

    rows = []
    for _, brand_row in brand_replies.iterrows():
        parent_id = brand_row["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id not in tweets_by_id.index:
            continue
        customer_row = tweets_by_id.loc[parent_id]
        if isinstance(customer_row, pd.DataFrame):  # duplicate ids, be defensive
            customer_row = customer_row.iloc[0]
        if customer_row["inbound"] is not True and str(customer_row["inbound"]).lower() != "true":
            continue  # parent wasn't actually a customer message

        customer_text = customer_row["text"]
        brand_text = brand_row["text"]
        if len(customer_text) < 5 or len(brand_text) < 5:
            continue

        # grab one turn of prior context if it exists (customer's earlier msg)
        prior_context = ""
        grandparent_id = customer_row.get("in_response_to_tweet_id")
        if pd.notna(grandparent_id) and grandparent_id in tweets_by_id.index:
            gp = tweets_by_id.loc[grandparent_id]
            if isinstance(gp, pd.DataFrame):
                gp = gp.iloc[0]
            prior_context = clean_text(gp.get("text", ""))

        rows.append({
            "thread_id": customer_row["tweet_id"],
            "prior_context": prior_context,
            "customer_message": customer_text,
            "brand_reply": brand_text,
            "created_at": customer_row.get("created_at", ""),
        })
        if len(rows) >= max_threads:
            break

    out = pd.DataFrame(rows).drop_duplicates(subset=["customer_message", "brand_reply"])
    log.info("Reconstructed %d clean customer<->%s reply pairs.", len(out), brand)
    return out


def main():
    df = load_raw()
    threads = build_brand_threads(df, config.BRAND, config.MAX_BRAND_THREADS)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    threads.to_parquet(config.BRAND_THREADS_PARQUET, index=False)
    log.info("Saved -> %s", config.BRAND_THREADS_PARQUET)


if __name__ == "__main__":
    main()
