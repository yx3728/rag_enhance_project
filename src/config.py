"""Shared config for the pilot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
EVAL = DATA / "eval"
INDEX = DATA / "index"
RESULTS = ROOT / "results"
for d in (RAW, EVAL, INDEX, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# Models (Claude via the `claude` CLI, OAuth/Max account).
# Answering model: used for BOTH base-only and RAG generation in the crown-jewel ablation,
# so the only thing that changes is whether retrieved context is supplied. A capable-but-small
# model makes the "small model + RAG" product story concrete and keeps full-eval-set cost low.
ANSWER_MODEL = "claude-haiku-4-5"
# Judge: a stronger, different model than the answerer -> reduces self-preference bias.
JUDGE_MODEL = "claude-opus-4-8"

# Local embedding model (sentence-transformers, CPU). Free + reproducible recall@k.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval defaults
TOP_K = 5
CHUNK_SIZE = 800       # chars (matches enterprise-copilot default)
CHUNK_OVERLAP = 120
