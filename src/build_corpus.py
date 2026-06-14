"""Generic corpus builder for the multi-repo scale-up (reuses the dagster diagnosis fixes:
heading-aware/code-safe chunking, MDX cleanup, API-docstring extraction).

Per-repo config in REPOS: which doc dirs + extensions, which source packages to mine for API
docstrings (the real API reference, since Sphinx/mkdocstrings rST/md are autodoc stubs), and
whether to expand dagster-style <CodeExample> directives (dagster only).

Writes data/index/<key>/chunks.json. Embeddings are built lazily by the metric scripts.

Usage: python src/build_corpus.py <repo_key>
"""
import json
import re
import sys

import config as C
from chunking import Chunk
from corpus import chunk_markdown, clean_mdx, expand_code_examples
import apidocs

REPOS = {
    "duckdb__duckdb": dict(
        ref="_ref_duckdb-web",
        doc_dirs=[("docs/current", (".md",))],
        apidoc_pkgs=[], expand_ce=False),
    "litestar-org__litestar": dict(
        ref="_ref_litestar",
        doc_dirs=[("docs", (".rst", ".md"))],
        apidoc_pkgs=["litestar"], expand_ce=False),
    "pydantic__pydantic-ai": dict(
        ref="_ref_pydantic-ai",
        doc_dirs=[("docs", (".md", ".mdx"))],
        apidoc_pkgs=["pydantic_ai_slim/pydantic_ai"], expand_ce=False),
}


def clean_rst(text: str) -> str:
    # strip orphan/layout fields + autodoc directives; keep prose/code
    text = re.sub(r"^:(\w[\w-]*):.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\.\. (currentmodule|module|automodule|autoclass|autofunction|autodata|autoattribute|toctree|include|seealso|figure|image)::.*$",
                  "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build(repo_key: str):
    cfg = REPOS[repo_key]
    ref = C.ROOT / cfg["ref"]
    chunks: list[Chunk] = []
    n_docs = 0

    for rel_dir, exts in cfg["doc_dirs"]:
        base = ref / rel_dir
        for fp in sorted(base.rglob("*")):
            if fp.suffix.lower() not in exts:
                continue
            rel = f"{rel_dir}/{fp.relative_to(base)}"
            raw = fp.read_text(encoding="utf-8", errors="replace")
            if fp.suffix.lower() == ".rst":
                text = clean_rst(raw)
            else:
                text = clean_mdx(expand_code_examples(raw) if cfg["expand_ce"] else raw)
            if len(text) < 40:
                continue
            chunks += chunk_markdown(rel, text)
            n_docs += 1

    if cfg["apidoc_pkgs"]:
        for rel, text in apidocs.extract(ref, package_dirs=cfg["apidoc_pkgs"]):
            chunks += chunk_markdown(rel, text)
            n_docs += 1

    out_dir = C.INDEX / repo_key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(json.dumps(
        [{"chunk_id": c.chunk_id, "doc_path": c.doc_path, "chunk_index": c.chunk_index,
          "content": c.content, "heading": c.heading} for c in chunks], indent=2))
    import statistics
    lens = [len(c.content) for c in chunks]
    api = sum(1 for c in chunks if c.doc_path.startswith("api/"))
    print(f"built '{repo_key}': {n_docs} docs -> {len(chunks)} chunks "
          f"(api-docstring chunks={api}) | median {int(statistics.median(lens))} chars, "
          f"tiny<120 {sum(1 for l in lens if l < 120)}, max {max(lens)}")


if __name__ == "__main__":
    build(sys.argv[1])
