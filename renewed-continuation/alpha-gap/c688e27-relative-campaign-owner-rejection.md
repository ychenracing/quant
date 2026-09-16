# Relative campaign owner: fixed paired-screen rejection

## Decision

`relative_campaign_owner` is `REJECTED_PAIRED_SCREEN`. The fixed attribution
state was genuinely predictive in the unchanged control account, but using it
as the complete mature-campaign continuation/exit authority changed subsequent
cash, admissions and acquisition bases. The resulting dynamic path regressed in
two of three scopes and increased turnover in all three.

No 2026 window or final acceptance matrix was run. Do not search the fifth/tenth
review sessions, CSI 300 benchmark, sign conditions, confirmation days,
fractions, capacity or per-pool variants. The common-five gain cannot be spliced
into another candidate.

## Causal read-only attribution that justified the screen

The diagnostic classified actual funded campaigns at fixed owned-session 5 and
10 closes. A quality campaign had positive stock-minus-CSI-300 excess log return
at both reviews and positive absolute stock return at the full review. Campaigns
closed before review or still open at the frozen cutoff could not make it pass.

| Scope | Quality / other campaigns | Win rate quality / other | Median PnL quality / other | Positive PnL retained | Negative PnL in other |
|---|---:|---:|---:|---:|---:|
| union | 14 / 17 | 78.57% / 52.94% | CNY 169.7k / 28.6k | 74.11% | 53.43% |
| common-five | 8 / 12 | 100% / 50.00% | CNY 567.3k / 6.9k | 80.53% | 100% |
| leader removal | 15 / 18 | 73.33% / 44.44% | CNY 56.3k / -40.1k | 65.82% | 76.73% |

The state passed every rule fixed before implementation. Attribution result
SHA256 is
`1069c361c6529163d1a004f2dad49b0c9a93ba83ddf222316d0f1435660175a5`;
diagnostic source SHA256 is
`b761729b9d2f5c60739e4bfc28a7f0a384dd3447dde1c43f88ea9f7a398b6d7b`.

## Preregistered implementation

The preregistration was preserved at research commit
`5675004346ce448acd517d0cdf3f6208e8931bb1`, tree
`01f44ae22fbe7deef5e7135da42d03279e45c658`, before implementation or account
measurement. Initial admission, ranking, two-position capacity and funding were
unchanged. Actual first fills started the campaign clock. Before session 10 the
existing price-book exits remained authoritative; after maturity the fixed
relative state owned continuation and exit. Missing data, an 8% one-session
loss, account risk, existing inventory obligations and execution constraints
remained authoritative.

## Fixed paired results

| Scope | Wealth control -> treatment | Relative change | MDD change | Orders change | Gross-turnover change |
|---|---:|---:|---:|---:|---:|
| union | 4.744148x -> 4.349091x | -8.33% | +1.34pp | +14 | +14.87 |
| common-five | 7.904183x -> 8.109177x | +2.59% | +0.47pp | +8 | +13.90 |
| leader removal | 1.154679x -> 0.669727x | -42.00% | +16.68pp | +28 | +20.71 |

Transition counts (activated/rejected/revoked) were 22/24/6, 17/17/5 and
19/26/10. The result therefore fails the fixed no-regression rule. Maximum
ledger reconciliation error was `5.996e-9` CNY.

## Verification

- focused tests: 6/6 passed after an observed red phase with 6 expected missing-
  implementation failures;
- ordinary tests: 87/87 passed;
- research tests: 420 passed, 9 skipped;
- repeated screen output was byte-identical, SHA256
  `1952e8849495510affb3dc9212d64b44635f21057097afbe0ecf4e7908c6b60e`.

## Continuation constraint

This result separates prediction from actionability: a state can describe
eventual winners in the unchanged account yet fail when it changes the account's
future opportunity path. Direct benchmark-relative lifecycle replacement is
closed. Future alpha work must identify a decision whose treatment does not
create repeated exit/re-entry feedback, and must not repackage this state as an
entry gate, retention wrapper, funding tilt, latch, sleeve or pool-specific
splice.
