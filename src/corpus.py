"""Phase-1 mechanical corpus build for dagster.

Fixes the obvious defects found in the baseline corpus:
  1. CORPUS SCOPE — adds the Sphinx API reference (docs/sphinx/**/*.rst), which the baseline
     (docs/docs/ only) omitted; usage/API questions often answer there.
  2. CODEEXAMPLE STUBS — 24% of baseline chunks contained unexpanded `<CodeExample path=.../>`
     directives. We expand each by reading the referenced snippet (relative to examples/) and
     extracting the region between its `# start_X` / `# end_X` markers, inlined as a code fence.
  3. MDX NOISE — strip frontmatter, MDX import lines, `{/* */}` comments; turn `<PyObject .../>`
     into the bare object name.
  4. CHUNKING — heading-aware splitting that never breaks a fenced code block (vs the baseline's
     blind 800-char windows that split code/tables mid-block).

Reads from the shallow clone at _ref_dagster/. Writes chunks for a new index key so the baseline
index is preserved for comparison.

Usage: python src/corpus.py            # builds tool key dagster-io__dagster_mech
"""
import json
import re
from pathlib import Path

import config as C
from chunking import Chunk

REF = C.ROOT / "_ref_dagster"
DOCS = REF / "docs" / "docs"
SPHINX = REF / "docs" / "sphinx"
EXAMPLES = REF / "examples"

OUT_TOOL = "dagster-io__dagster_mech"

CODEEXAMPLE_RE = re.compile(r"<CodeExample\b([^>]*?)/>", re.DOTALL)
ATTR_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|\{([^}]*)\})')
PYOBJECT_RE = re.compile(r"<PyObject\b([^>]*?)/>", re.DOTALL)
MDX_IMPORT_RE = re.compile(r"^import\s+.*?from\s+['\"].*?['\"];?\s*$", re.MULTILINE)
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
SELF_CLOSING_JSX_RE = re.compile(r"<(Tabs|TabItem|Note|Warning|details|summary)\b[^>]*>", re.IGNORECASE)


def _attrs(s: str) -> dict:
    out = {}
    for m in ATTR_RE.finditer(s):
        out[m.group(1)] = m.group(2) or m.group(3) or m.group(4) or ""
    return out


def load_snippet(path: str, start_after: str, end_before: str) -> str | None:
    """path is relative to examples/. Return the marked region (or whole file)."""
    fp = EXAMPLES / path
    if not fp.exists():
        return None
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    s, e = 0, len(lines)
    if start_after:
        for i, ln in enumerate(lines):
            if start_after in ln:
                s = i + 1
                break
    if end_before:
        for i in range(s, len(lines)):
            if end_before in lines[i]:
                e = i
                break
    region = lines[s:e]
    # drop other marker comment lines and blank edges
    region = [ln for ln in region if not re.match(r"\s*#\s*(start|end)_", ln)]
    while region and not region[0].strip():
        region.pop(0)
    while region and not region[-1].strip():
        region.pop()
    if not region:
        return None
    # simple dedent
    indents = [len(ln) - len(ln.lstrip()) for ln in region if ln.strip()]
    d = min(indents) if indents else 0
    region = [ln[d:] if len(ln) >= d else ln for ln in region]
    lang = "python" if path.endswith(".py") else (path.rsplit(".", 1)[-1] if "." in path else "")
    return f"```{lang}\n" + "\n".join(region) + "\n```"


def expand_code_examples(text: str) -> str:
    def repl(m):
        a = _attrs(m.group(1))
        path = a.get("path", "")
        snip = load_snippet(path, a.get("startAfter", ""), a.get("endBefore", ""))
        if snip is None:
            return f"```\n# (code example: {path})\n```"
        return snip
    return CODEEXAMPLE_RE.sub(repl, text)


def clean_mdx(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    text = MDX_COMMENT_RE.sub("", text)
    text = MDX_IMPORT_RE.sub("", text)

    def pyobj(m):
        a = _attrs(m.group(1))
        return f"`{a.get('object') or a.get('module') or a.get('decorator') or ''}`"
    text = PYOBJECT_RE.sub(pyobj, text)
    text = SELF_CLOSING_JSX_RE.sub("", text)
    text = re.sub(r"</(Tabs|TabItem|Note|Warning|details|summary)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- heading-aware, code-fence-safe chunking ----------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_sections(text: str):
    """Split into (heading, body) sections at H1-H3 boundaries; keep code fences intact."""
    lines = text.split("\n")
    sections, cur_head, cur = [], "", []
    in_fence = False
    for ln in lines:
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
        m = HEADING_RE.match(ln) if not in_fence else None
        if m and len(m.group(1)) <= 3:
            if cur:
                sections.append((cur_head, "\n".join(cur).strip()))
            cur_head, cur = m.group(2).strip(), [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append((cur_head, "\n".join(cur).strip()))
    return [(h, b) for h, b in sections if b]


def _split_blocks(body: str):
    """Split a section body into blocks (paragraphs / whole code fences)."""
    blocks, buf, in_fence = [], [], False
    for ln in body.split("\n"):
        if ln.lstrip().startswith("```"):
            if in_fence:
                buf.append(ln); blocks.append("\n".join(buf)); buf = []; in_fence = False
            else:
                if buf:
                    blocks.append("\n".join(buf)); buf = []
                buf.append(ln); in_fence = True
        elif in_fence:
            buf.append(ln)
        elif ln.strip() == "":
            if buf:
                blocks.append("\n".join(buf)); buf = []
        else:
            buf.append(ln)
    if buf:
        blocks.append("\n".join(buf))
    return [b for b in blocks if b.strip()]


def chunk_markdown(doc_path: str, text: str, target: int = 1100, hard_max: int = 2200) -> list[Chunk]:
    """Heading-aware: each chunk is prefixed with its heading; sections packed to ~target chars;
    code fences never split (a fence larger than hard_max becomes its own chunk)."""
    out, idx = [], 0
    for heading, body in _split_sections(text):
        prefix = f"## {heading}\n" if heading else ""
        blocks = _split_blocks(body)
        buf = ""
        for blk in blocks:
            cand = (buf + "\n\n" + blk).strip() if buf else blk
            if len(cand) <= target or not buf:
                buf = cand
                # a single oversized block (huge code fence): emit alone
                if len(buf) > hard_max:
                    out.append(Chunk(f"{doc_path}#{idx}", doc_path, idx, (prefix + buf).strip(), heading)); idx += 1
                    buf = ""
                continue
            out.append(Chunk(f"{doc_path}#{idx}", doc_path, idx, (prefix + buf).strip(), heading)); idx += 1
            buf = blk
        if buf.strip():
            out.append(Chunk(f"{doc_path}#{idx}", doc_path, idx, (prefix + buf).strip(), heading)); idx += 1
    return _merge_small(doc_path, out, target)


def _merge_small(doc_path: str, chunks: list[Chunk], target: int) -> list[Chunk]:
    """Merge consecutive chunks within a doc while the combined size stays <= target.
    Folds tiny heading-only/short sections into their neighbours so they aren't isolated."""
    if not chunks:
        return chunks
    merged: list[str] = []
    buf = chunks[0].content
    for c in chunks[1:]:
        if len(buf) + len(c.content) + 2 <= target:
            buf = buf + "\n\n" + c.content
        else:
            merged.append(buf); buf = c.content
    merged.append(buf)
    # enforce a hard cap so nothing exceeds what the embedder can encode (~512 tok ≈ 2200 chars)
    capped: list[str] = []
    for content in merged:
        if len(content) <= 2200:
            capped.append(content); continue
        lines, cur = content.split("\n"), []
        for ln in lines:
            cur.append(ln)
            if len("\n".join(cur)) >= 1600:
                capped.append("\n".join(cur)); cur = []
        if cur:
            capped.append("\n".join(cur))
    out = []
    for i, content in enumerate(capped):
        head = ""
        first = content.lstrip().split("\n", 1)[0]
        m = HEADING_RE.match(first) if first.startswith("#") else None
        if m:
            head = m.group(2).strip()
        out.append(Chunk(f"{doc_path}#{i}", doc_path, i, content.strip(), head))
    return out


def build():
    chunks: list[Chunk] = []
    n_docs = 0
    # 1. prose docs (md/mdx) with CodeExample expansion + MDX cleanup
    for fp in sorted(DOCS.rglob("*")):
        if fp.suffix.lower() not in (".md", ".mdx"):
            continue
        rel = "docs/docs/" + str(fp.relative_to(DOCS))
        raw = fp.read_text(encoding="utf-8", errors="replace")
        text = clean_mdx(expand_code_examples(raw))
        if len(text) < 40:
            continue
        chunks += chunk_markdown(rel, text)
        n_docs += 1
    # 2. Sphinx API reference (.rst) — light cleanup, same chunker
    for fp in sorted(SPHINX.rglob("*.rst")):
        rel = "docs/sphinx/" + str(fp.relative_to(SPHINX))
        raw = fp.read_text(encoding="utf-8", errors="replace")
        # strip rst directive noise lightly; keep text
        text = re.sub(r"^\.\. (currentmodule|module|autoclass|autofunction|autodata)::.*$", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 40:
            continue
        chunks += chunk_markdown(rel, text)
        n_docs += 1

    out_dir = C.INDEX / OUT_TOOL
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(json.dumps(
        [{"chunk_id": c.chunk_id, "doc_path": c.doc_path, "chunk_index": c.chunk_index,
          "content": c.content, "heading": c.heading} for c in chunks], indent=2))
    stub = sum(1 for c in chunks if "<CodeExample" in c.content)
    tiny = sum(1 for c in chunks if len(c.content) < 120)
    print(f"built corpus '{OUT_TOOL}': {n_docs} docs -> {len(chunks)} chunks "
          f"| remaining CodeExample stubs={stub} | tiny(<120)={tiny}")
    return chunks


if __name__ == "__main__":
    build()
