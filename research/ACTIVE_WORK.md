# Current quant continuation

Original task and PR #1 remain `NOT_COMPLETE / NOT_MET`. Alpha-first authority
continues: maximize return capability without changing frozen data, costs,
execution, evidence identity or acceptance. Preserve all rejected candidates;
no defensive patching, score/funding micro-tuning, neighboring searches,
per-pool splicing or reference-source reuse.

## Live identities

- main: `966f7428658431c9da5a6fcaf7d9b9d5c446f2cd`;
- PR #1: OPEN, Draft, unmerged; implementation head
  `efffb9a428258e921feaff125011cba46223bb18`;
- previous research status: `889574dc6efba364bdb6ef03a2871969ce0e645e`;
- frozen pre-2026 data:
  `894230b20361fb826d58b27987e87146a090b9e894e7abf091121486f5293157`.

## Latest alpha closure

The fixed `accelerating_persistence` acquisition state is
`REJECTED_BEFORE_IMPLEMENTATION`. It used only the first-buy signal close and
unchanged fast/slow horizons: latest 10-session log-return velocity greater than
the preceding 30-session velocity, with both positive. Actual settled campaign
results for state=true were:

- union: 24 campaigns, 37.50% wins, median cash PnL CNY -60,827;
- common-five: 17 campaigns, 47.06% wins, median CNY -17,050;
- leader removal: 25 campaigns, 36.00% wins, median CNY -22,138.

The state retained only 22.61%/19.23%/42.07% of positive campaign cash PnL and
did not have a higher win rate in any scope. Do not tune or repackage its
horizons, velocity relation or any individual scope.

Two negative controls also failed. Loss-following reentry contained some of the
largest later winners (CNY +0.980m union, +5.073m common-five, +1.352m removal).
Fresh false-to-true transitions of the unchanged entry predicate produced total
cash PnL CNY -0.684m/-0.413m/-0.077m, materially below already-live trend
admissions in all scopes. Thus acquisition gates would remove important alpha;
they do not explain the shared lifecycle gap.

Source/result SHA256 are
`0e3ab86b25f933278dad93f8271b42575b3ae6eccbe80beb2827a181035e3935` /
`38e5df8de2b121db308370b060b15c677688f5b5ec7be094eaaac92cc305dd7a`.

Do not repeat `campaign_payoff`, `joint_funding`, `confirmed_shock`,
inventory/T+1 corrections, `profit_trend_shield`, theme/leader slots,
handover/reentry, persistent/early ownership, breadth/sector ownership,
accelerator, campaign epoch, participation, reversal, record-high, breakout,
rotation, acquisition acceleration/fresh-start/loss-memory gates or
decision-path grids.

## Direct continuation

Do not implement another acquisition gate. The next independent work must use
the same actually settled campaigns to compare prefix-observable state *after*
ownership begins. Identify one security-specific deterioration event that
separates adverse broad-scope campaigns from sustained common-five compounders
before any complete owner is preregistered. The event must be causal, must not
depend on eventual return, ticker, supplied-pool branch or reference source, and
must show cross-scope coverage of negative PnL without truncating the large
later winners. No risk wrapper, exit-horizon grid or neighboring threshold
search is allowed.

Writer state: released after this additive save. No accepted production tree or
pending economic run exists. PR #1 remains Draft/unmerged.
