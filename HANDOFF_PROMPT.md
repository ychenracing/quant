# HANDOFF_PROMPT — quant current recovery entry

Updated 2026-09-15 after the workbuddy/common-five T+1 execution reconciliation. This continues `ychenracing/quant` and original PR #1; it is not a new project. Read `renewed-continuation/PRIOR_HANDOFF.md` and original goal anchor `b54eaaf6487005a492b23ee5bce300f93f7e41cf:HANDOFF_PROMPT.md` when full historical authority is needed. Read repository instructions, `docs/RESEARCH_CONTRACT.md`, current source contracts, `renewed-continuation/progress.json`, and exact source/evidence identities. Live GitHub refs and newer explicit user instructions override stale snapshots.

## Current authority and stopping rules

The original three decision-first economic hypotheses completed and were rejected; `decision-review/budget.json` remains immutable and closed. The later renewed continuation was prospectively bounded to two substantively different fixed economic hypotheses. Both completed and were rejected. Registered/measured/rejected: 2; pending: 0; remaining: 0. Do not reset either ledger, run a third economic variant, tune neighbors, combine rejected mechanisms, or rename a failed idea as a new hypothesis under the closed budgets.

Correctness-only execution/reference reconciliation remains independently authorized and does not consume/reopen the economic research budget. Preserve original and corrected reference reports separately. Do not publish private reference source, use another strategy codebase as quant's implementation base, or treat reference repair as production economic acceptance.

Original economic acceptance remains `NOT_MET`; production delivery remains `NOT_COMPLETE`. No verified final production candidate currently satisfies the full original contract. PR #1 stays Draft/unmerged until the same production candidate satisfies all applicable original economic, correctness and repository requirements. Do not reduce thresholds or infer mathematical impossibility from rejected local hypotheses.

## Verified Git and data identities

- main: `bbc002449093e458868a8748687df023958ba283` at the latest verified recovery snapshot; re-read live main before writes/merge.
- original PR #1: OPEN, Draft, unmerged; head `research/independent-tech@4fa306fdcde2f9a77d941200ef7ff2f988c045e5` at latest verified snapshot.
- research implementation: `research/expectation-learning@32304314270586d078d527aa2860aab78a09f613`, tree `253804819b1d43c17c066eb1e353f40c3eaa762a`.
- parent capacity implementation: `7825767424b17db6afb1ed3b5e660b0316c7a383`, tree `50a423cc112e45b56d6b24a7d222d5560472137d`.
- frozen data: 34 names, 896 sessions, 2023-01-03..2026-09-11; full SHA256 `d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b`. No pre-2023 warmup or appended prices.

Never treat a synthetic/local recovery Git ancestry as remote source ancestry. A production-package hash is not research identity. Re-read live refs and collaboration state before modifying shared branches or dispatching runs.

## Completed economic experiments — recover, do not dispatch again

1. `opportunity_allocation`: contract `80c0252aa3f1bb33fb4b158209521f14aed43dcf`; source `7825767424b17db6afb1ed3b5e660b0316c7a383`; run `34925189658`; evidence `nonlinear/7825767424b17db6afb1ed3b5e660b0316c7a383/34925189658/`; REJECTED; archive SHA256 `3c7ae6af6fd16c01840f53f16801d4d6672058e06deba15d957b1c7767a998a5`; 51 manifest members verified.

2. `continuity_selection`: contract `5418e5f5be67d6caffa5a60581eda1730e88579f`; source `32304314270586d078d527aa2860aab78a09f613`; run `34926804636`; evidence `nonlinear/32304314270586d078d527aa2860aab78a09f613/34926804636/`; REJECTED; archive SHA256 `316471d9b036f3c4e6cfc0899a1e59858fcd8c353db8d52267f062da838170be`; 63 manifest members verified.

Both used the existing unmodified paired screen. Neither rejected treatment was promoted into a final full-period acceptance matrix. Do not splice per-pool winners. Preserve all failures and the earlier retained controls source-bound.

## Latest correctness result — workbuddy/common-five T+1 reconciliation

This item is now measured and closed for the workbuddy/common-five reference. Do not repeat it merely because an older note says corrected wealth was unknown.

Authenticated control/reference identity:
- private reference artifact: `10023909983`
- archive source SHA: `c3d838ce6349f8bdf0ec2988836212563aaa10bc`
- reference `quant_ai.py` SHA256: `d74e8cd6c9a68823a08c5ed5526d72e65dce7b9cdc7568eeb38d1f9d80483a29`
- control wealth: `20.221072996623576x`
- control MDD: `25.3319520590%`
- control trade rows: 107
- control trades SHA256: `a9620cef60d7f18623af3d676ac7aaaa29d7de9613712812764998eecb406adb`

Independent complete-ledger audit found exactly one T+1 violation: on 2025-04-28 `sz300502`, beginning sellable inventory was zero, 52,100 units were bought that day and then 52,100 were sold by `dd_hard_limit` later the same session.

An isolated private comparator changed only execution-inventory semantics: sells consume beginning-session sellable inventory net of prior same-day sales; same-day buys/top-ups do not replenish same-day sellable quantity; blocked protective quantity is deferred to the next available session; delayed hard-breaker stand-down completes only when the deferred action actually completes. Private reference/corrected source was not published; only source hashes and resulting evidence are preserved.

Corrected fixed-account result:
- T+1 violations: 0
- 52,100-unit `sz300502 dd_hard_limit` sale moved from 2025-04-28 to 2025-04-29
- wealth: `20.81972538210324x`, +`2.9605371860%` relative versus control
- MDD: `25.2953421603%`, about `0.03661` percentage points lower
- first 57 pre-divergence trade rows exact
- equity curve exact through 2025-04-27
- independent cash replay closes to reported final equity within floating-point tolerance and ends flat
- generic old-inventory/top-up, repeated-attempt, partial-fill, next-session-retry and fresh-inventory-lock semantics: PASS

Substantive evidence checkpoint: `research/evidence@d9c6b3bc0f089ea482336bfa1f6c1b86d9934a9b`, tree `ce3f3a93cb14ed5fb2e866482f8e580f0250f801`, path `renewed-continuation/native-execution/`. Raw measurement archive SHA256 `dc75f8962f13c994c01e5038860a3ad03f7580abbf9af7371fd2365af7b76741`; reconstruct it from the eight base64 parts using the checked manifest.

Interpretation is bounded: this proves the identified T+1 error did not inflate the workbuddy common-five headline; in this fixed comparison the corrected account improves slightly. It does **not** normalize intraday information timing, fee/liquidity semantics, corporate actions, broader risk-state semantics, actual-share accounting, or the other three private references. It is not quant production acceptance; corrected MDD still exceeds the original 18% ceiling.

## Remaining independently actionable work

Continue correctness-only restoration of a comparable four-reference execution basis **only from authenticated raw identities/evidence**. For Qwen/GLM/Codex, first locate exact source/artifact/run identity and raw outputs already preserved; do not infer execution semantics from remembered headline wealth. Build an explicit matrix of what is authenticated, what is comparable, and what remains missing. If a reference cannot be authenticated from preserved evidence, record that exact gap rather than reconstructing source from memory or rerunning an unbounded experiment.

Do not repeat workbuddy T+1 measurement. Do not dispatch completed economic pairs. Do not reset the research budget. A genuinely new economic mechanism needs new verifiable premise plus prospective boundaries/authorization, not a renamed tuning loop.

Original requirements remain in force: independent quant implementation; one production entry; cash long/no leverage/no short/no auto orders; after-close decision and next-trading-day execution; original eligibility/risk limits/cash/T+1/sellable inventory/protection retries/fees/liquidity/nonresettable actual-NAV risk; original pools/common-five/union/contributor and industry removals/subset sizes/full and bull periods/cost-delay/neighborhood/prefix; both 2026-06-22..08-31 and 2026-07-01..08-31. No uquant/trade 15x, 40-order, or 10%-waiver rules apply.

If the same final production candidate eventually meets every applicable original economic requirement, correctness requirement, and live repository protection, normally merge original PR #1 and verify main without requesting redundant approval. Until then, keep PR #1 Draft/unmerged and clearly distinguish engineering correctness, economic acceptance, and production completion.
