"""Probe candidate repos: do they have GitHub Discussions Q&A with accepted answers?

Uses `gh api graphql` (authenticated). Prints a table so we can shortlist substrates.
"""
import json
import subprocess
import sys

CANDIDATES = [
    ("astral-sh", "uv"), ("pola-rs", "polars"), ("astral-sh", "ruff"),
    ("duckdb", "duckdb"), ("tauri-apps", "tauri"), ("pydantic", "pydantic"),
    ("PrefectHQ", "prefect"), ("fastapi", "typer"), ("marimo-team", "marimo"),
    ("Textualize", "textual"), ("dagster-io", "dagster"), ("encode", "httpx"),
    ("sveltejs", "kit"), ("withastro", "astro"), ("oven-sh", "bun"),
    ("bigskysoftware", "htmx"), ("fastapi", "fastapi"), ("langchain-ai", "langchain"),
    ("tiangolo", "sqlmodel"), ("Lightning-AI", "pytorch-lightning"),
    ("python-poetry", "poetry"), ("pdm-project", "pdm"), ("astral-sh", "rye"),
    ("strawberry-graphql", "strawberry"), ("litestar-org", "litestar"),
    ("modal-labs", "modal-client"), ("streamlit", "streamlit"),
    ("gradio-app", "gradio"), ("plotly", "dash"), ("apache", "airflow"),
]

Q = """
query($owner:String!,$name:String!){
  repository(owner:$owner,name:$name){
    nameWithOwner
    hasDiscussionsEnabled
    stargazerCount
    answered: discussions(first:1, answered:true){ totalCount }
  }
}
"""


def probe(owner, name):
    try:
        out = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={Q}",
             "-F", f"owner={owner}", "-F", f"name={name}"],
            capture_output=True, text=True, timeout=30,
        )
        d = json.loads(out.stdout)
        r = d.get("data", {}).get("repository")
        if r is None:
            return (f"{owner}/{name}", None, None, None)
        return (r["nameWithOwner"], r["hasDiscussionsEnabled"],
                r["answered"]["totalCount"], r["stargazerCount"])
    except Exception as e:
        return (f"{owner}/{name}", f"ERR {e}", None, None)


rows = [probe(o, n) for o, n in CANDIDATES]
# sort: discussions enabled + answered count desc
rows.sort(key=lambda x: (x[1] is True, x[2] or 0), reverse=True)
print(f"{'repo':32} {'discuss':8} {'answered_qa':12} {'stars':8}")
print("-" * 64)
for repo, disc, ans, stars in rows:
    print(f"{repo:32} {str(disc):8} {str(ans):12} {str(stars):8}")
