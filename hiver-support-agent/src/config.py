"""
Central configuration for the Hiver support-agent pipeline.

Everything that a grader / reproducer might want to change lives here or in
.env — nothing else in the codebase should hardcode a brand name, model
name, or file path.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------
# The dataset (Kaggle: thoughtvector/customer-support-on-twitter) has dozens
# of brand handles in the `author_id` field of the *brand* (non-customer)
# tweets. We picked AmazonHelp for the reference run because it has one of
# the largest, cleanest volumes of resolved customer<->brand threads, which
# matters for the "grounded in how the brand has historically resolved
# similar issues" requirement. Change BRAND to re-target any other handle
# present in the raw CSV (e.g. AppleSupport, Uber_Support, SpotifyCares).
BRAND = os.getenv("HIVER_BRAND", "AmazonHelp")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
GOLDEN_DIR = ROOT_DIR / "data" / "golden_eval"

RAW_CSV = RAW_DIR / "twcs.csv"                       # the full Kaggle CSV
BRAND_THREADS_PARQUET = PROCESSED_DIR / f"{BRAND.lower()}_threads.parquet"
RETRIEVAL_INDEX_DIR = PROCESSED_DIR / f"{BRAND.lower()}_index"
GOLDEN_SET_CSV = GOLDEN_DIR / "golden_eval.csv"
GOLDEN_LABELED_CSV = GOLDEN_DIR / "golden_eval_labeled.csv"
EVAL_RESULTS_CSV = PROCESSED_DIR / "eval_results.csv"

# How many rows to keep from the (huge, ~3M row) raw file when subsampling.
# The assignment explicitly says a subsample is expected.
MAX_BRAND_THREADS = int(os.getenv("HIVER_MAX_THREADS", "8000"))

# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------
# Defined by reading ~150 AmazonHelp threads by hand (see decision_log.md,
# decision #2). Kept deliberately small (9 classes) — a huge taxonomy is
# easy to define and impossible to keep an LLM (or a human) consistent on.
INTENTS = [
    "delivery_delay",       # package late / stuck in transit / not delivered
    "order_cancellation",   # customer wants to cancel or already tried to
    "refund_request",       # money back, wrong charge, double charge
    "product_defect",       # item broken / wrong item / missing parts
    "account_access",       # login, password, locked account, 2FA
    "return_exchange",      # wants to send item back / swap for another
    "billing_inquiry",      # question about a charge, invoice, subscription
    "general_inquiry",      # how-to / status question, no clear problem yet
    "complaint_escalation", # angry, repeat contact, threatening to leave / already escalated
]

# Intents that are essentially never safe to auto-resolve without a human,
# regardless of what the classifier / retrieval says. Used by the escalation
# policy in src/agent.py.
ALWAYS_ESCALATE_INTENTS = {"complaint_escalation", "account_access"}

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
# Free-to-use options only (see README "Cost" section for why each was
# picked). Default is Ollama because it needs literally zero signup / API
# key / credit card — just a local install. Groq and Gemini both have
# generous free tiers if you'd rather not run a local model.
LLM_PROVIDER = os.getenv("HIVER_LLM_PROVIDER", "ollama")  # ollama | groq | gemini

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Judge can be pointed at a different (ideally stronger) free model than the
# agent itself, to reduce self-grading bias. Defaults to same provider.
JUDGE_LLM_PROVIDER = os.getenv("HIVER_JUDGE_PROVIDER", LLM_PROVIDER)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Local, free, no API key: sentence-transformers running on CPU.
EMBEDDING_MODEL = os.getenv("HIVER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RETRIEVAL = int(os.getenv("HIVER_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Escalation confidence threshold
# ---------------------------------------------------------------------------
# Below this intent-classification confidence, auto-escalate regardless of
# intent (see decision_log.md, decision #7).
INTENT_CONFIDENCE_ESCALATION_THRESHOLD = float(
    os.getenv("HIVER_CONF_THRESHOLD", "0.55")
)

RANDOM_SEED = 42
