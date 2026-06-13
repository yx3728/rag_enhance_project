"""Build the eval set for a chosen tool (spec §3).

- Chunk the tool's docs corpus, build the index (cached).
- For each answered discussion, locate gold doc chunk(s) via answer->corpus embedding match
  (the answer is a *different* query than the question we later retrieve on). A question is
  kept (docs-answerable + grounded) iff its top answer->corpus similarity >= TAU and it passes
  basic quality filters.
- Persist:
    data/index/<tool>/chunks.json       (corpus chunks)
    data/index/<tool>/emb.npz            (embedding cache)
    data/eval/<tool>.json                (eval set)

Usage: python src/build_eval.py <owner__name> [TAU]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

import config as C
from chunking import chunk_document, Chunk
from index import Index, embed_texts

TAU = 0.62          # min answer->corpus cosine to count a chunk as gold / keep the question
MAX_GOLD = 3        # cap gold chunks per question
MIN_Q_LEN = 40
MIN_ANS_LEN = 120   # drop trivial "fixed in next release" answers


def load_corpus(tool: str) -> list[Chunk]:
    docs_dir = C.RAW / "docs" / tool
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in (".md", ".mdx") or path.name == "_manifest.json":
            continue
        rel = str(path.relative_to(docs_dir))
        text = path.read_text(encoding="utf-8", errors="replace")
        text = strip_frontmatter(text)
        if len(text.strip()) < 50:
            continue
        chunks += chunk_document(rel, text, strategy="paragraph",
                                 size=C.CHUNK_SIZE, overlap=C.CHUNK_OVERLAP)
    return chunks


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text


def question_text(item: dict) -> str:
    return f"{item['title']}\n\n{item['question']}".strip()


def build(tool: str, tau: float = TAU):
    chunks = load_corpus(tool)
    if not chunks:
        raise SystemExit(f"No docs found for {tool} under {C.RAW/'docs'/tool}. Run fetch_docs first.")
    print(f"corpus: {len(chunks)} chunks from docs")

    idx_dir = C.INDEX / tool
    idx_dir.mkdir(parents=True, exist_ok=True)
    (idx_dir / "chunks.json").write_text(json.dumps(
        [{"chunk_id": c.chunk_id, "doc_path": c.doc_path, "chunk_index": c.chunk_index,
          "content": c.content, "heading": c.heading} for c in chunks], indent=2))
    idx = Index.build(chunks, cache_path=idx_dir / "emb.npz")

    discussions = json.load(open(C.RAW / "discussions" / f"{tool}.json"))
    # embed all accepted answers at once (answer -> corpus matching)
    answers = [d["answer"] for d in discussions]
    ans_emb = embed_texts(answers)            # (M, d) normalized
    sims = ans_emb @ idx.emb.T                # (M, N)

    eval_rows = []
    kept_top_sims = []
    for d, srow in zip(discussions, sims):
        if len(question_text(d)) < MIN_Q_LEN or len(d["answer"].strip()) < MIN_ANS_LEN:
            continue
        order = np.argsort(-srow)
        top = order[:MAX_GOLD]
        gold = [(idx.ids[i], float(srow[i])) for i in top if srow[i] >= tau]
        if not gold:
            continue
        kept_top_sims.append(float(srow[order[0]]))
        eval_rows.append({
            "id": f"{tool}#{d['number']}",
            "number": d["number"],
            "title": d["title"],
            "question": question_text(d),
            "reference_answer": d["answer"],
            "url": d["url"],
            "gold_chunk_ids": [cid for cid, _ in gold],
            "gold_top_sim": round(gold[0][1], 3),
        })

    out_path = C.EVAL / f"{tool}.json"
    out_path.write_text(json.dumps({
        "tool": tool, "tau": tau, "n_corpus_chunks": len(chunks),
        "n_discussions": len(discussions), "n_eval": len(eval_rows),
        "questions": eval_rows,
    }, indent=2))
    print(f"eval set: kept {len(eval_rows)}/{len(discussions)} questions (tau={tau}) -> {out_path}")
    if kept_top_sims:
        arr = np.array(kept_top_sims)
        print(f"  kept top-sim: min={arr.min():.3f} median={np.median(arr):.3f} max={arr.max():.3f}")


if __name__ == "__main__":
    tool = sys.argv[1]
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else TAU
    build(tool, tau)
