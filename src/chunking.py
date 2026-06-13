"""Chunking — ported near-verbatim from enterprise-copilot/backend/app/services/chunking.py.

Char-based fixed / paragraph / sentence strategies. Pure functions, no external deps.
We add a `source` tag so chunks carry provenance (which doc + section heading) for recall@k
ground-truth mapping.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str        # stable id: "<doc_path>#<index>"
    doc_path: str        # source doc (relative path within the docs corpus)
    chunk_index: int
    content: str
    heading: str = ""    # nearest markdown heading, for human-readable provenance


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _merge_parts(parts: list[str], max_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}\n\n{part}".strip() if buffer else part
        if len(candidate) <= max_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(part) <= max_size:
            buffer = part
        else:
            chunks.extend(_fixed_chunks(part, max_size, overlap))
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_text(text: str, *, strategy: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if strategy == "paragraph":
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return _merge_parts(parts, size, overlap)
    if strategy == "sentence":
        sentences = re.split(r"(?<=[.!?])\s+", text)
        parts = [s.strip() for s in sentences if s.strip()]
        return _merge_parts(parts, size, overlap)
    return _fixed_chunks(text, size, overlap)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _nearest_heading(text: str, pos: int) -> str:
    """Best-effort: last markdown heading at or before `pos`."""
    last = ""
    for m in _HEADING_RE.finditer(text):
        if m.start() > pos:
            break
        last = m.group(2).strip()
    return last


def chunk_document(doc_path: str, text: str, *, strategy: str, size: int, overlap: int) -> list[Chunk]:
    pieces = chunk_text(text, strategy=strategy, size=size, overlap=overlap)
    out: list[Chunk] = []
    search_from = 0
    for i, piece in enumerate(pieces):
        # locate the piece to attach a nearby heading (best effort)
        idx = text.find(piece[:40], search_from) if piece else -1
        heading = _nearest_heading(text, idx) if idx >= 0 else ""
        if idx >= 0:
            search_from = idx
        out.append(Chunk(
            chunk_id=f"{doc_path}#{i}",
            doc_path=doc_path,
            chunk_index=i,
            content=piece,
            heading=heading,
        ))
    return out
