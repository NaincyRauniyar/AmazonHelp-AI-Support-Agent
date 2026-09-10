"""
Embedding-based retrieval over historical (customer_message -> brand_reply)
pairs for the target brand. This is what "grounds" the drafted reply in how
the brand has actually resolved similar issues before, rather than letting
the LLM freelance a policy it wasn't trained on.

Local + free: sentence-transformers (open weights, runs on CPU) + a flat
numpy cosine-similarity index. No vector DB needed at this data scale
(thousands, not millions, of threads per brand after subsampling).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

_model_cache = {}


def _get_embedder():
    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s (first call only)...", config.EMBEDDING_MODEL)
        _model_cache["model"] = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model_cache["model"]


class ThreadRetriever:
    """Wraps a brand's resolved-thread table with a nearest-neighbor index
    over customer_message embeddings."""

    def __init__(self, threads: pd.DataFrame):
        self.threads = threads.reset_index(drop=True)
        self._embeddings: np.ndarray | None = None

    def build(self):
        embedder = _get_embedder()
        texts = self.threads["customer_message"].tolist()
        log.info("Embedding %d historical customer messages...", len(texts))
        self._embeddings = embedder.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        return self

    def save(self, path):
        path = str(path)
        np.save(path + "_emb.npy", self._embeddings)
        self.threads.to_parquet(path + "_meta.parquet", index=False)

    @classmethod
    def load(cls, path):
        path = str(path)
        threads = pd.read_parquet(path + "_meta.parquet")
        retriever = cls(threads)
        retriever._embeddings = np.load(path + "_emb.npy")
        return retriever

    def query(self, message: str, top_k: int = None) -> list[dict]:
        if self._embeddings is None:
            raise RuntimeError("Call .build() or .load() first")
        top_k = top_k or config.TOP_K_RETRIEVAL
        embedder = _get_embedder()
        q = embedder.encode([message], normalize_embeddings=True)[0]
        sims = self._embeddings @ q  # cosine sim, since both are normalized
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            row = self.threads.iloc[idx]
            results.append({
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
                "similarity": float(sims[idx]),
            })
        return results


def build_and_save_index():
    """CLI entry point: python -m src.retrieval"""
    threads = pd.read_parquet(config.BRAND_THREADS_PARQUET)
    retriever = ThreadRetriever(threads).build()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(config.RETRIEVAL_INDEX_DIR)
    log.info("Saved retrieval index -> %s", config.RETRIEVAL_INDEX_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_save_index()
