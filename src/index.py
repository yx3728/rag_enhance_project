"""In-process retrieval index — replaces enterprise-copilot's Qdrant *server* with a
numpy cosine index (same search semantics: cosine top-k over chunk embeddings), plus a
BM25 lexical index and a hybrid combiner. Small corpus -> no daemon, fully reproducible.

Dense embeddings come from a local sentence-transformers model (free, offline). Embeddings
are cached to disk keyed by (model, content hash) so re-runs are instant.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import config as C
from chunking import Chunk

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(C.EMBED_MODEL)
    return _model


def embed_texts(texts: list[str], *, is_query: bool = False) -> np.ndarray:
    """Embed texts -> L2-normalized float32 matrix. bge models want a query prefix."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    inp = texts
    if is_query and "bge" in C.EMBED_MODEL.lower():
        inp = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    vecs = model.encode(inp, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _tok(s: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9_]+", s.lower())


class Index:
    """Holds chunks + dense embeddings + BM25; supports vector / bm25 / hybrid search."""

    def __init__(self, chunks: list[Chunk], emb: np.ndarray):
        self.chunks = chunks
        self.emb = emb  # (N, d) normalized
        self.ids = [c.chunk_id for c in chunks]
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tok(c.content) for c in chunks])

    # ---- builders ----
    @classmethod
    def build(cls, chunks: list[Chunk], cache_path: Path | None = None) -> "Index":
        contents = [c.content for c in chunks]
        emb = None
        if cache_path and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            if list(data["ids"]) == [c.chunk_id for c in chunks]:
                emb = data["emb"]
        if emb is None:
            emb = embed_texts(contents)
            if cache_path:
                np.savez(cache_path, ids=np.array([c.chunk_id for c in chunks]), emb=emb)
        return cls(chunks, emb)

    # ---- search ----
    def vector(self, query: str, k: int) -> list[tuple[str, float]]:
        q = embed_texts([query], is_query=True)[0]
        scores = self.emb @ q  # cosine (all normalized)
        idx = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tok(query))
        idx = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def hybrid(self, query: str, k: int, alpha: float = 0.5, pool: int = 50) -> list[tuple[str, float]]:
        """Reciprocal-rank-style fusion via min-max normalized score blend over a candidate pool."""
        vec = dict(self.vector(query, pool))
        bm = dict(self.bm25(query, pool))

        def norm(d: dict[str, float]) -> dict[str, float]:
            if not d:
                return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            rng = (hi - lo) or 1.0
            return {kk: (vv - lo) / rng for kk, vv in d.items()}

        nvec, nbm = norm(vec), norm(bm)
        keys = set(nvec) | set(nbm)
        fused = {kk: alpha * nvec.get(kk, 0.0) + (1 - alpha) * nbm.get(kk, 0.0) for kk in keys}
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return ranked

    def retrieve(self, query: str, k: int, method: str = "vector") -> list[tuple[str, float]]:
        return {"vector": self.vector, "bm25": self.bm25, "hybrid": self.hybrid}[method](query, k)

    def chunk_by_id(self, cid: str) -> Chunk:
        return self.chunks[self.ids.index(cid)]


def content_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode()); h.update(c.content.encode())
    return h.hexdigest()[:12]
