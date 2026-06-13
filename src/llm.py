"""LLM client for the pilot — calls Claude headlessly via the `claude` CLI.

No API key required: runs on the user's logged-in OAuth/Max account. Pattern adapted from
~/cog_moral_tests/run_experiment.py. We use `--output-format json` (one JSON object per call)
with "clean" flags so the call behaves like a bare API request (empty system prompt, no tools,
no MCP, no settings, no session persistence).

Every call's cost (`total_cost_usd`) and token usage are returned so the pilot can sum spend and
track the LLM-judge budget.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

# Clean flags: make `claude -p` behave like a bare model call.
_CLEAN_FLAGS = [
    "--tools", "",
    "--strict-mcp-config",
    "--setting-sources", "",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--output-format", "json",
]

# Env vars to scrub so nothing perturbs the call / forces an API-key path.
_SCRUB = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_ENTRYPOINT",
    "MAX_THINKING_TOKENS",
    "CLAUDE_CODE_SIMPLE",
]

PER_CALL_TIMEOUT_S = 240


@dataclass
class LLMResult:
    text: str
    model: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    is_error: bool = False
    error: str = ""


@dataclass
class UsageTracker:
    """Sums cost + counts calls, with a hard cap on judge calls (the pilot's ≤500 budget)."""
    judge_budget: int = 500
    gen_calls: int = 0
    judge_calls: int = 0
    total_cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def record(self, r: LLMResult, *, is_judge: bool) -> None:
        if is_judge:
            self.judge_calls += 1
        else:
            self.gen_calls += 1
        self.total_cost_usd += r.cost_usd
        m = self.by_model.setdefault(r.model, {"calls": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["cost_usd"] += r.cost_usd

    def judge_remaining(self) -> int:
        return self.judge_budget - self.judge_calls

    def summary(self) -> dict:
        return {
            "gen_calls": self.gen_calls,
            "judge_calls": self.judge_calls,
            "judge_budget": self.judge_budget,
            "judge_remaining": self.judge_remaining(),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "by_model": {k: {"calls": v["calls"], "cost_usd": round(v["cost_usd"], 4)}
                         for k, v in self.by_model.items()},
        }


def call_claude(
    prompt: str,
    *,
    model: str = "claude-opus-4-8",
    system: str = "",
    effort: str | None = None,
    max_retries: int = 4,
) -> LLMResult:
    """Single headless Claude call. Returns an LLMResult (never raises on model/CLI error)."""
    argv = ["claude", "-p", prompt, "--model", model, "--system-prompt", system]
    if effort:
        argv += ["--effort", effort]
    argv += _CLEAN_FLAGS

    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}

    last_err = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            proc = subprocess.run(
                argv, env=env, capture_output=True, text=True,
                timeout=PER_CALL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {PER_CALL_TIMEOUT_S}s"
            continue
        dur = int((time.time() - t0) * 1000)

        if proc.returncode != 0:
            last_err = f"exit {proc.returncode}: {proc.stderr.strip()[:400]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        try:
            obj = json.loads(proc.stdout)
        except json.JSONDecodeError:
            last_err = f"unparseable stdout: {proc.stdout.strip()[:300]}"
            continue

        if obj.get("is_error"):
            last_err = f"model error: {obj.get('result', '')[:300]}"
            time.sleep(1.5 * (attempt + 1))
            continue

        return LLMResult(
            text=(obj.get("result") or "").strip(),
            model=model,
            cost_usd=float(obj.get("total_cost_usd") or 0.0),
            input_tokens=int((obj.get("usage") or {}).get("input_tokens") or 0),
            output_tokens=int((obj.get("usage") or {}).get("output_tokens") or 0),
            duration_ms=int(obj.get("duration_ms") or dur),
        )

    return LLMResult(text="", model=model, is_error=True, error=last_err)


if __name__ == "__main__":
    r = call_claude("Reply with exactly: PONG", model="claude-haiku-4-5")
    print(r)
