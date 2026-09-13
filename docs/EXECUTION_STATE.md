# Execution state

Status: active implementation; economics NOT accepted.

## Authorized scope

Independently implement a cash-only, long-only, daily A-share technology decision-support system. Reference the four projects in `ychenracing/trades` read-only; do not fork a strategy, copy reference functions, or inherit a reference engine. Work from 2023-01-03 only, including indicator warm-up. Preserve code and research incrementally on `research/independent-tech`; publish a truthful usable delivery to main without a further approval request. No automatic orders and no claim of universal superiority without evidence.

## Recovery point verified on 2026-09-13

- main: `234cc5729455ce5440b0a5307aadbcb9c1c0bbc4` (LICENSE only).
- resumed branch: `research/independent-tech`, previous HEAD `3fd96a75f34ce47ce7bbfd751784130162eaa145`.
- Existing research contract and acquisition scripts retained. Other branch `research/technology-strategy` is preserved, not rewritten.
- No AGENTS.md exists in either main or the resumed branch at these revisions.
- Local direct Git access cannot resolve github.com; the authorized GitHub connector supports repository reads and is used for writes.

## Available evidence

Uploaded archives: `trades-engineering-source-evidence.zip`, `quant-market-evidence.zip`, `quant-market-supplement.zip`, `quant-public-data-evidence.zip`, `gquant-economic-evidence.zip`, `uquant-frozen-data-evidence.zip`.
The first supplies read-only comparator source. The primary market archive supplies raw and adjusted OHLCV for the technology union, plus observation indices. Complete regular histories run 2023-01-03 to 2026-09-11 (896 sessions); IPOs and suspensions have shorter histories. The original Tencent `bj920045` response has only one bar; the explicitly supplied Eastmoney supplement has 170 bars starting 2025-12-31. Do not silently treat shortened histories as full coverage. Corporate-action accounting and point-in-time universe provenance must be audited before any execution-grade economic claim.

## Implementation plan

1. Audit data hashes, coverage, adjustments and the four reference entry points; record baseline identities and limitations.
2. Write an independent small package separating validated data, causal trend selection, asymmetric risk budgeting/recovery, next-session execution, and evidence/CLI. One global parameter set, no symbol/date alpha switches.
3. Check critical invariants with a compact offline suite: causal prefixes, cash/position conservation, next-session timing, blocked fills, risk-exit priority and deterministic evidence.
4. Run original-pool/leader-removal/subset/cost/regime diagnostics and native comparators where reproducible. Preserve all candidates and failures; never substitute proxy returns for native acceptance.
5. Publish architecture/parameters/operations/results/limitations and exact source/data/runtime identity; run short validation, push and read back the resulting branch and main SHAs.

No strategy implementation or economic test has yet been completed at this checkpoint. This document will be updated with actual results rather than used as evidence of completion.
