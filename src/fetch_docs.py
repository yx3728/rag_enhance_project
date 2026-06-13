"""Fetch a tool's official docs (markdown) from its GitHub repo via the git trees API.

Downloads .md/.mdx files under the given path prefixes, caches raw text to
data/raw/docs/<tool>/ preserving relative paths. This is the RAG corpus.

Usage: python src/fetch_docs.py <owner> <name> <ref> <path_prefix>[,<path_prefix>...]
  e.g. python src/fetch_docs.py duckdb duckdb main docs/
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import config as C

DOCS_RAW = C.RAW / "docs"


def gh_json(path: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=90)
        if out.returncode == 0:
            return json.loads(out.stdout)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gh api {path} failed: {out.stderr[:200]}")


def fetch(owner: str, name: str, ref: str, prefixes: list[str],
          exts=(".md", ".mdx")) -> int:
    tree = gh_json(f"repos/{owner}/{name}/git/trees/{ref}?recursive=1")
    blobs = [t for t in tree["tree"]
             if t["type"] == "blob"
             and t["path"].lower().endswith(exts)
             and any(t["path"].startswith(p) for p in prefixes)]
    out_dir = DOCS_RAW / f"{owner}__{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for b in blobs:
        dest = out_dir / b["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            n += 1
            continue
        blob = gh_json(f"repos/{owner}/{name}/git/blobs/{b['sha']}")
        content = base64.b64decode(blob["content"]) if blob.get("encoding") == "base64" else blob["content"].encode()
        dest.write_bytes(content)
        n += 1
        if n % 25 == 0:
            print(f"  ... {n}/{len(blobs)} files")
            time.sleep(0.5)
    meta = {"owner": owner, "name": name, "ref": ref, "prefixes": prefixes, "n_files": n}
    (out_dir / "_manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"{owner}/{name}: {n} doc files -> {out_dir}")
    return n


if __name__ == "__main__":
    owner, name, ref, prefixes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    fetch(owner, name, ref, prefixes)
