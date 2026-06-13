"""Fetch answered GitHub Discussions (Q&A) for a repo via `gh api graphql`.

For each answered discussion we keep: number, title, body (the question), the accepted
answer body, url, category, createdAt. Cached to data/raw/discussions/<owner>__<name>.json
so we don't re-fetch.

Usage: python src/fetch_discussions.py <owner> <name> [max]
"""
import json
import subprocess
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "discussions"
RAW.mkdir(parents=True, exist_ok=True)

Q = """
query($owner:String!,$name:String!,$cursor:String){
  repository(owner:$owner,name:$name){
    discussions(first:50, after:$cursor, answered:true,
                orderBy:{field:UPDATED_AT, direction:DESC}){
      pageInfo{ hasNextPage endCursor }
      nodes{
        number title url createdAt
        category{ name }
        body
        answer{ body url }
      }
    }
  }
}
"""


def _graphql(args: list[str], max_retries: int = 6) -> dict:
    """Run a gh graphql call, retrying on null-data / errors / secondary rate limits."""
    import time
    for attempt in range(max_retries):
        out = subprocess.run(args, capture_output=True, text=True, timeout=90)
        try:
            d = json.loads(out.stdout) if out.stdout.strip() else {}
        except json.JSONDecodeError:
            d = {}
        if d.get("data", {}).get("repository") is not None:
            return d
        # error envelope or rate limit -> back off
        wait = 5 * (attempt + 1)
        msg = (d.get("errors") or out.stderr or "null data")
        print(f"  retry {attempt+1}/{max_retries} in {wait}s ({str(msg)[:120]})", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"graphql failed after {max_retries} retries: {out.stdout[:200]} {out.stderr[:200]}")


def fetch(owner: str, name: str, max_items: int = 200) -> list[dict]:
    import time
    items: list[dict] = []
    cursor = None
    while len(items) < max_items:
        args = ["gh", "api", "graphql", "-f", f"query={Q}",
                "-F", f"owner={owner}", "-F", f"name={name}"]
        if cursor:
            args += ["-F", f"cursor={cursor}"]
        d = _graphql(args)
        disc = d["data"]["repository"]["discussions"]
        for n in disc["nodes"]:
            if not n.get("answer"):
                continue
            items.append({
                "number": n["number"],
                "title": n["title"],
                "url": n["url"],
                "created_at": n["createdAt"],
                "category": (n.get("category") or {}).get("name"),
                "question": n["body"],
                "answer": n["answer"]["body"],
                "answer_url": n["answer"]["url"],
            })
        if not disc["pageInfo"]["hasNextPage"]:
            break
        cursor = disc["pageInfo"]["endCursor"]
        time.sleep(1.0)  # be gentle to avoid secondary rate limits
    return items[:max_items]


def main():
    owner, name = sys.argv[1], sys.argv[2]
    max_items = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    items = fetch(owner, name, max_items)
    path = RAW / f"{owner}__{name}.json"
    path.write_text(json.dumps(items, indent=2))
    print(f"{owner}/{name}: fetched {len(items)} answered discussions -> {path}")


if __name__ == "__main__":
    main()
