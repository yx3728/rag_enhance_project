"""Shared loaders for the eval harnesses."""
import json

import config as C
from chunking import Chunk
from index import Index


def load_index(tool: str) -> Index:
    idx_dir = C.INDEX / tool
    raw = json.load(open(idx_dir / "chunks.json"))
    chunks = [Chunk(**c) for c in raw]
    return Index.build(chunks, cache_path=idx_dir / "emb.npz")


def load_eval(tool: str) -> dict:
    return json.load(open(C.EVAL / f"{tool}.json"))
