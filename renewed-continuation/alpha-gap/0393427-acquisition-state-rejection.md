# Acquisition-state attribution: rejected before implementation

This read-only diagnostic used the already authenticated incumbent accounts,
their actual filled first-buy/settlement cash flows, and frozen market history
through 2025-12-31.  It did not replay a treatment, use reference source code,
change a threshold, or use an open/right-censored campaign as a settled label.

## Fixed hypothesis

The only candidate state was registered before measurement and inherited the
unchanged `Config.fast=10` and `Config.slow=40` horizons:

`log(P[t]/P[t-10])/10 > log(P[t-10]/P[t-40])/30 > 0`.

Economically, the current ten-session price velocity must exceed an already
positive preceding thirty-session velocity.  This tests accelerating persistent
ownership rather than another score, funding, breadth or risk threshold.

The state failed every scope's advance rule:

| scope | state campaigns | win rate | median cash PnL | state positive PnL / all positive PnL | state negative PnL |
|---|---:|---:|---:|---:|---:|
| union | 24 | 37.50% | CNY -60,827 | 22.61% | CNY -2.182m |
| common-five | 17 | 47.06% | CNY -17,050 | 19.23% | CNY -1.460m |
| leader removal | 25 | 36.00% | CNY -22,138 | 42.07% | CNY -1.032m |

The complementary campaigns had win rates of 40.00%, 47.37% and 40.00%, and
carried more positive PnL in every scope.  The state therefore neither selected
compounding nor removed most adverse churn.  It is
`REJECTED_BEFORE_IMPLEMENTATION`; do not tune either horizon, velocity boundary,
or combine a scope row with another candidate.

Two causal negative controls further rule out convenient relabelings:

- Loss-following reentry was not the bad state. It contained union's later
  `300308` CNY +0.980m campaign, common-five's `300502` CNY +5.073m campaign,
  and removal's `601869` CNY +1.352m campaign.
- A fresh false-to-true transition of the unchanged production entry predicate
  was worse than admission during an already-live trend: its aggregate cash PnL
  was CNY -0.684m / -0.413m / -0.077m across union/common-five/removal, while
  non-fresh campaigns produced CNY +6.000m / +8.399m / +1.916m.  Fresh-start
  win rates were only 29.41% / 44.44% / 31.25%.

These controls are descriptive exclusions, not replacement candidates.  They
show that the shared deficit cannot be repaired by another acquisition gate:
the largest gains often arrive after a prior loss or inside an already mature
trend.  The remaining independent question is lifecycle state after actual
ownership begins: whether a prefix-observable, security-specific deterioration
event can end adverse broad-scope campaigns without truncating the mature
common-five compounders.  Any continuation must first quantify that event on
the same closed campaigns; no gate, leader wrapper, risk wrapper, named stock,
pool branch, or nearby threshold search is authorized.

Source SHA256: `0e3ab86b25f933278dad93f8271b42575b3ae6eccbe80beb2827a181035e3935`  
Result SHA256: `38e5df8de2b121db308370b060b15c677688f5b5ec7be094eaaac92cc305dd7a`
