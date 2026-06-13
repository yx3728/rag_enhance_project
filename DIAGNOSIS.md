# RAG Diagnosis — dagster (before any scale-up)

**Task.** The dagster RAG pilot's numbers were weak (reported recall@5 = 0.67; crown-jewel lift
only +6.6 after the fallback fix). Hypothesis to test: *architecture/implementation problem, not an
inherent task ceiling.* Method: clear obvious mechanical defects → re-measure → forensically triage
the residual into corpus-gap / metric-artifact / chunking / embedding → fix only what the residual
justifies → re-run the crown jewel. Cost was unconstrained but logged (~$70 Opus total).

## TL;DR

The weak numbers were **mostly a measurement artifact plus a corpus-coverage limit — not a
retrieval-architecture ceiling.**

1. **The reported recall@5 = 0.673 was an illusion.** The labeled "gold" chunks were largely
   content-empty `<CodeExample path=.../>` **stubs** (24% of all chunks). Retrieval found the stub
   (recall looked fine) but the stub had no answer text. A gold-independent **answer-coverage@5**
   judge put honest retrieval at **0.18** on the same corpus.
2. **But that strict coverage was itself an artifact** in the other direction: judged against
   verbose human reference answers it had a ~30% false-negative rate. A **lenient claim-coverage**
   metric (does any retrieved chunk *support* the answer, partial/reworded OK) puts real coverage@5
   at **~0.53**.
3. **Forensic triage of the residual:** ~49% corpus-gap (answer is community/debugging knowledge
   not in the docs), ~30% metric-artifact, ~12% chunking, ~9% embedding-rank. Only **~17–22% is
   retrieval-architecture-fixable.**
4. **Fixes that helped:** expanding `<CodeExample>` stubs to real code + heading-aware chunking
   (corrected the corpus and the gold/recall artifact); adding the real API reference (all 71
   libraries' source **docstrings**, since the Sphinx rST is just autodoc stubs) lifted claim-
   coverage@5 0.527 → **0.567** and cut corpus-gap 0.247 → 0.213.
5. **Fixes that didn't:** an off-the-shelf cross-encoder reranker **hurt** (cov@5 0.567 → 0.48) on
   these long, messy discussion-style queries.

**Conclusion:** the ceiling here is **corpus coverage + the eval's reliance on community answers**,
not the embedding/retriever architecture. This should reshape the multi-repo plan (see end).

## The numbers (n=150, same questions throughout)

| stage | config | exact recall@5 | strict coverage@5 | lenient claim-coverage@5 | corpus-gap |
|---|---|---|---|---|---|
| **baseline (frozen)** | docs/docs only, 800-char chunks | **0.673** (inflated) | **0.18** | — | — |
| **post-mechanical** | + CodeExample-expanded, Sphinx, heading-aware chunks | 0.54¹ | 0.20 | **0.527** | 0.247 |
| **post-advanced** | **+ all-package API docstrings (`_wide`)** | 0.507¹ | — | **0.567** | **0.213** |
| (rejected lever) | `_wide` + bge-reranker-base | — | — | 0.48 ↓ | 0.213 |

¹ exact recall@5 is measured against gold *remapped* to the re-chunked corpus (lexical overlap),
and the baseline 0.673 was inflated by stub-gold — so exact-recall is **not** comparable across
rows. The honest, gold-independent comparable is **claim-coverage@5**.

## What each metric tells us

- **recall@5 (exact-chunk):** unreliable here. Inflated at baseline by stub gold; sensitive to
  re-chunking. Use it only within a fixed corpus/gold.
- **strict answer-coverage@5 (whole-reference match):** robust to gold mislabel, but it inherits the
  LLM-judge **verbosity bias** — against terse-but-correct retrieved chunks vs verbose community
  references it under-scores by ~30% (measured: 36 of 120 "misses" actually had the answer in
  top-5). Good for detecting the stub problem; too harsh as the headline.
- **lenient claim-coverage@5 (any chunk supports the answer):** the honest headline retrieval metric
  here. ~0.53 (`_mech`) → ~0.57 (`_wide`).

## Residual triage (forensic, on the 120 strict-coverage misses)

| code | category | n | % residual | retrieval-fixable? |
|---|---|---|---|---|
| (a) | corpus gap — answer not in top-20 whole-corpus (community/debug/roadmap knowledge) | 59 | 49% | No |
| (b) | metric artifact — answer WAS in top-5; strict judge false-negative | 36 | 30% | No (already retrieved) |
| (c) | chunking — present but fragmented across chunks | 14 | 12% | Partly |
| (d) | embedding rank — good single chunk exists, ranked ≥ k | 11 | 9% | Yes |

Examples — **(a)**: "what does gRPC `UNAVAILABLE` mean" (debugging knowledge, not in docs);
"use dynamic partitions, refresh with a sensor or smth" (conversational). **(b)**: a question whose
answer chunk was retrieved at **rank 0** but the strict judge marked "not covered" because the
community reference added extra specifics. **(c)/(d)**: answer exists in a doc/docstring chunk but
ranked below 5 or split across chunks.

## Mechanical vs advanced contribution (separable)

- **Mechanical (corpus hygiene):** the big win was *correctness of measurement*, not a coverage
  jump — it exposed that recall@5=0.67 was fake (stub gold) and that the real retrieval was ~0.53
  (lenient). It also expanded 1,152 stub chunks to real code and fixed code/table-splitting chunks.
- **Advanced (corpus widening):** +4 pts claim-coverage@5 and −3.4 pts corpus-gap from adding the
  real API reference (1,035 source-docstring modules). Modest because most residual gap is
  community knowledge no docstring contains. Reranker: net-negative here.

## Crown jewel — RAG vs base, best config vs original

<!-- CROWN -->
_(filled in after the `_wide` strict+fallback ablation completes)_

## Implementation vs real ceiling

- **Implementation problems (fixed):** stub-poisoned corpus + stub-poisoned gold (inflated recall),
  blind char-chunking, a missing API reference, and a too-strict coverage metric. These explain
  most of the "weak numbers."
- **Real ceiling (not implementation):** ~21% of questions are answered only by community/debugging
  knowledge that isn't in the dagster docs at all, and the eval references are verbose human answers
  that no single doc chunk reproduces verbatim. Better embeddings/rerankers cannot fix either.

## Implications for the multi-repo plan

- **Lead with measurement.** Use lenient claim-coverage (pooled, partial-credit) as the retrieval
  metric; never trust exact-chunk recall@k when gold is auto-labeled (stub/fragment risk).
- **Budget for corpus coverage, not just retrieval tuning.** Expect a substantial fraction of
  forum/Discussion questions to be community knowledge absent from docs — measure the corpus-gap
  rate per repo up front; it bounds achievable RAG quality more than the retriever does.
- **Mechanical corpus hygiene first** (expand include-directives, real API docstrings, structural
  chunking). **Defer rerankers/embedding swaps** — here they were ≤ the fixable ~17% and the
  off-the-shelf reranker hurt; only invest if a repo's triage shows a large (c)+(d) slice.
