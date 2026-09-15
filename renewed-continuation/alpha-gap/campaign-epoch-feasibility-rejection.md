# Standalone campaign-epoch feasibility rejection

## Question

After wrapper-level admission, retention, handover, reentry, breadth and sector
ownership all failed, test whether a complete independent market-level campaign
epoch can supply sparse launches and persistent ownership across the three core
scopes. This is a read-only feasibility screen. It does not consume reference
strategy source, modify a portfolio policy, or run a candidate account.

## Fixed causal definition

Using frozen data through 2025-12-31:

- a security is ready after 60 observations and at least 80% active sessions in
  the trailing 20;
- it is qualified when close > EMA20 > EMA60 and 60-session return is positive;
- an epoch launches when at least half of ready names qualify, the causal
  supplied-universe equal-weight index EMA20 exceeds EMA60, and the median
  60-session return is positive;
- it finishes only after all three conditions reverse;
- the launch cohort is measured from the next open to the next-open finish with
  unchanged commission and slippage, no membership changes and no leverage.

Advance required every scope to have at least two settled epochs, positive
median settled net return, at least half positive, and settled compound factor
above one. Open/censored campaigns are recorded but excluded.

## Result

| Scope | Launches | Settled | Median settled net | Positive settled | Settled compound |
|---|---:|---:|---:|---:|---:|
| union | 5 | 4 | -7.17% | 0/4 | 0.7651x |
| common-five | 4 | 3 | +20.10% | 2/3 | 1.1413x |
| joint optical-leader removal | 5 | 4 | -6.81% | 0/4 | 0.7690x |

The final 2025 epoch remained open and marked +112.45%/+191.86%/+82.94% across
the three scopes. Those marks are censored. They cannot offset four completed
losses in union and removal or be promoted as executable acceptance. Every
launch stayed within the unchanged 0.5% ADV entry-capacity diagnostic.

The mechanism therefore identifies the late-2025 boom but fails to explain
cross-scope alpha. Status is `REJECTED_BEFORE_IMPLEMENTATION`. No strategy,
2026 window, native comparison or final matrix was run. Do not tune the
majority threshold, EMA/return windows, finish rule, cohort weights, or splice
the common-five and open-epoch rows.

## Verification and identity

- frozen selection data:
  `894230b20361fb826d58b27987e87146a090b9e894e7abf091121486f5293157`;
- source SHA256:
  `211b46c53b2749bb2b4aaa43f1cd9a3528b409e6f7af4ff0fe66295a831316d6`;
- result SHA256:
  `dc8fa4e78a60664626b03e4bbb3a4a2e501c565930cfff78df97741c072c6d4c`;
- ordinary tests: 87/87;
- research tests: 410 passed, 9 skipped.

The diagnostic uses adjusted economic units and is not verified actual-share,
corporate-action or tax accounting. All history is retrospective.

## Continuation

Do not implement a campaign-epoch owner. Before any next implementation,
attribute the already saved same-engine buy-hold and quant accounts to determine
whether a fixed causal admission or lifecycle event can improve winner drift in
union, common-five and removal. Named leaders, per-pool branches, extra wrappers
and neighboring searches remain prohibited.
