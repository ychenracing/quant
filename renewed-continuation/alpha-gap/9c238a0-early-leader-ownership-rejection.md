# Early leader ownership: fixed rejection and alpha-source reset

## Outcome

The fixed `early_leader_ownership` pair was registered before implementation,
implemented with tests first, and measured only on the three unchanged
pre-2026 core scopes. It is `REJECTED_PAIRED_SCREEN`.

| scope | control wealth | treatment wealth | change | control MDD | treatment MDD |
|---|---:|---:|---:|---:|---:|
| union | 4.7441484288x | 4.1189037362x | -13.1793% | 30.5579% | 30.6873% |
| common-five | 7.9041830487x | 5.6676404289x | -28.2957% | 39.1356% | 39.9688% |
| joint optical-leader removal | 1.1546791489x | 1.2189637440x | +5.5673% | 35.3310% | 37.8196% |

No treatment 2026 or final matrix was run. The candidate may not be retuned by
changing the ten-session observation clock, existing capacity rank, positive
gain sign, or retained fraction.

## What was tested

An actual holding could become confirmed only during the first unchanged
`Config.rebalance=10` observed closes after its first actual fill. Confirmation
required a positive close-to-weighted-acquisition-open gain, unchanged parent
admission and unchanged score rank within the parent's existing two-position
capacity. Confirmation never bought or enlarged units. Confirmed units could
survive market-only risk reductions; security brokenness, portfolio drawdown or
warning, and every pre-existing inventory obligation remained authoritative.

The retrospective persistent-account diagnosis that authorized this one test
had adequate activation and consistent labels: confirmed/unconfirmed median
cycle returns were +6.20%/-9.99%, +9.01%/-10.46% and +4.19%/-8.90% across the
three scopes. The economic pair shows why those labels were not a portfolio
counterfactual: selective retention changed later cash, admission and ownership
paths and lost substantial wealth in the two strongest scopes.

## Correctness and identity

- registration: `eacb68ea382b864b4630cb515a2277e6e77d43a2`
- expected RED test commit: `f75f33728bf3c1e62277ac48efe4d93bf2fa3efa`
- implementation: `dd39d42c01c935d8eee345bce97782244c4ccf72`
- rejection/result record: `9c238a0dd6ae0073cfdb385821ce7c94571f6f53`
- result tree: `2d6a1085e5f1fd46302be420d39b510e3e83c9cc`
- implementation SHA-256: `1845a86219dd731f7b3930acfd7ce49d75583b65fd0da07eafbd9b848e54312f`
- test SHA-256: `91610be627b24685c53967252942408b7f5674aa7dbfca595aba3de55f593b7a`
- contract SHA-256: `1530d99ff8811f7865f854b1049ad31a45afc74352fac8170eec8829d1183462`
- focused tests: 6/6; ordinary: 87/87; research: 393 passed, 9 skipped
- maximum treatment ledger residual: CNY3.93e-9
- matched hosted-control maximum error: CNY1.86e-9

## Direct continuation

Do not add another exit-retention wrapper to `trend_book`. The independently
saved broad buy-and-hold accounts already show that raw theme participation is
an alpha source: through 2025-12-31 wealth is 5.359743x / 19.027562x / 3.047536x
on union/common-five/removal, exceeding the current concentrated parent in all
three scopes. Separately, the already rejected persistent local-ownership
account was stronger on full-period union and removal but weaker on common-five.

The next diagnosis must therefore ask whether one prefix-causal market-breadth
state can choose the ownership *shape* inside a single standalone engine,
rather than choosing a different old candidate per pool. It must not rerun the
old `passive_guard`, `participation`, ranking, funding or exit-wrapper grids.
Only one new engine may be registered after confirming that its observable
state and failure screen are distinct from those closed families.

Economic acceptance remains `NOT_MET`; PR #1 stays Draft and unmerged.
