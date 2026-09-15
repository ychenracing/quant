# Causal theme-campaign fixed screen: rejected

Date: 2026-09-15 UTC
Source: `research/expectation-learning@69e56f375763ab78a94097ad05608ba03ece0ec1`
Source tree: `0e68627a83874f1bd43e89c17525a0732bd725eb`
Economic run: [35004916170](https://github.com/ychenracing/quant/actions/runs/35004916170)
Correctness run: [35004916094](https://github.com/ychenracing/quant/actions/runs/35004916094)

## Forward contract

The fixed treatment used the parent `trend_book.Owner` unchanged except for one
causal, generic-sector admission state.  An inactive state activated only when
the parent's two highest-ranked currently admissible names shared a sector.
The state persisted on the parent's existing 10-session review clock, and only
names in the active sector could receive increases.  There were no ticker,
date, sector, pool, threshold, cost, seed, universe, benchmark or execution
changes.  The registered pair was control `enabled=false` versus treatment
`enabled=true`.

The screen required wealth non-regression in all three scopes, at least one
strict wealth improvement, and a strict union improvement.  Failure stopped
the candidate before 2026 or final acceptance evaluation.

## Hosted fixed-pair result

| Scope | Control wealth | Treatment wealth | Relative change | Control MDD | Treatment MDD | Suppressed observations |
|---|---:|---:|---:|---:|---:|---:|
| union | 4.7441484288x | 1.9829668590x | -58.20% | 30.5579% | 34.1306% | 4,324 |
| common-five (`chatgpt_5`) | 7.9041830487x | 5.3737685430x | -32.01% | 39.1356% | 42.0254% | 396 |
| joint optical-leader removal | 1.1546791489x | 0.8106753497x | -29.79% | 35.3310% | 38.2653% | 3,954 |

Decision: `REJECTED_PAIRED_SCREEN`.

All three wealth results regressed and all three maximum drawdowns worsened.
No treatment 2026 run was made.  The evidence receipt says
`economic_acceptance=UNVERIFIED`; this result is not a production candidate.

Hosted archive SHA-256:
`6b81d5402bf0f84347898b9f44aacb856747ab2057a933cce0f9d82932376c1d`

Selection SHA-256:
`0402d763e62b4f51b147eccb6054fc63ac58491d9532f4288df7b90f0d98219e`

## Failure attribution

A read-only join of the treatment's causal suppression trace to the control
account's filled episode ledger shows why a hard theme veto is economically
wrong:

- union: 24 control episodes entered while the treatment suppressed the name;
  their aggregate realized PnL was **+1.271m**, with +3.361m winners and
  -2.091m losers.
- common-five: 10 such episodes produced **+9.477m** aggregate PnL.  One
  `sz300502` episode signalled 2025-05-09 and realized **+8.429m**
  (+253.14% on buy notional).
- leader-removal: 32 such episodes produced -0.360m in aggregate, but still
  contained +1.334m of winning PnL.

This is retrospective diagnostic attribution, not incremental acceptance
evidence.  It demonstrates that complete same-sector synchronization is too
sparse: the binary gate removes major isolated-leader alpha together with
distractors.

## Closed direction and next constraint

Do not tune the state persistence, top-two requirement, review interval or
nearby thresholds.  Do not repackage this candidate.  A subsequent structural
candidate may preserve the parent's strongest individual-leader admission and
use cohort evidence only for a marginal slot or lifecycle responsibility.
It must remain one causal rule across all pools and must be preregistered and
screened as a fresh fixed pair.
