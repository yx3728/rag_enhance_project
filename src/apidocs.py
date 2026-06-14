"""Extract public-API documentation from dagster source docstrings (the content the Sphinx
website renders via autodoc, which the raw .rst files do NOT contain).

For each .py module under the given packages, emit readable "API reference" docs: the module
docstring, and each public (non-underscore) class/function with its signature + docstring,
plus public methods of documented classes. Only objects WITH a docstring are kept (that is the
documented surface), which keeps signal high and noise low.

Returns a list of (doc_path, text) suitable for chunking by corpus.py.
"""
import ast
from pathlib import Path

# Core packages whose public docstrings form the SDK/API reference.
CORE_PKGS = [
    "python_modules/dagster/dagster",
    "python_modules/dagster-graphql/dagster_graphql",
    "python_modules/dagster-pipes/dagster_pipes",
]
# Skip non-API / internal packages when auto-discovering libraries.
SKIP_LIB = {"automation", "dagit", "dagster-test", "dagster-buildkite"}
SKIP_DIR_PARTS = {"tests", "test", "_tests"}


def discover_packages(ref_root: Path) -> list[str]:
    """All importable packages: core + every python_modules/libraries/<lib>/<lib_underscored>/."""
    globs = list(CORE_PKGS)
    libs = ref_root / "python_modules" / "libraries"
    if libs.exists():
        for d in sorted(libs.iterdir()):
            if not d.is_dir() or d.name in SKIP_LIB:
                continue
            pkg = d / d.name.replace("-", "_")
            if pkg.is_dir():
                globs.append(str(pkg.relative_to(ref_root)))
    return globs


def _sig(node) -> str:
    a = node.args
    parts = []
    posonly = getattr(a, "posonlyargs", [])
    for arg in posonly + a.args:
        parts.append(arg.arg)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    for arg in a.kwonlyargs:
        parts.append(arg.arg)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{node.name}({', '.join(parts)})"


def _module_doc(rel: str, tree: ast.Module) -> str | None:
    blocks = []
    mod_doc = ast.get_docstring(tree)
    if mod_doc and len(mod_doc) > 40:
        blocks.append(f"# Module {rel}\n\n{mod_doc}")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            blocks.append(f"## {_sig(node)}\n\n{doc}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            cdoc = ast.get_docstring(node)
            methods = []
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                    mdoc = ast.get_docstring(m)
                    if mdoc:
                        methods.append(f"### {node.name}.{_sig(m)}\n\n{mdoc}")
            if not cdoc and not methods:
                continue
            head = f"## class {node.name}\n\n{cdoc or ''}".strip()
            blocks.append("\n\n".join([head] + methods))
    if not blocks:
        return None
    return "\n\n".join(blocks)


def extract(ref_root: Path, package_dirs: list[str] | None = None) -> list[tuple[str, str]]:
    """Extract public-API docstring docs. If package_dirs is given (relative to ref_root), mine
    those; otherwise auto-discover dagster's packages (back-compat)."""
    docs = []
    globs = package_dirs if package_dirs is not None else discover_packages(ref_root)
    for glob in globs:
        base = ref_root / glob
        if not base.exists():
            continue
        for fp in base.rglob("*.py"):
            if any(part in SKIP_DIR_PARTS for part in fp.parts):
                continue
            if fp.name.startswith("_") and fp.name != "__init__.py":
                pass  # still allow; private filename can hold public API
            try:
                tree = ast.parse(fp.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            rel = "api/" + str(fp.relative_to(ref_root))
            text = _module_doc(rel, tree)
            if text and len(text) > 80:
                docs.append((rel, text))
    return docs


if __name__ == "__main__":
    import config as C
    docs = extract(C.ROOT / "_ref_dagster")
    total = sum(len(t) for _, t in docs)
    print(f"extracted API docstring docs: {len(docs)} modules, {total} chars")
    for rel, t in docs[:3]:
        print(f"\n--- {rel} ---\n{t[:300]}")
