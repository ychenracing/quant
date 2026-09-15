# Basis-failure exit: rejected paired screen

The contract was fixed remotely at commit `984f46f29e3bf691bc75d3d8a1092b765c64a886`
before implementation or treatment replay.  It defined one event only: while
actual units remain held, close below the observed weighted acquisition basis
and EMA40 for two consecutive completed sessions, then latch the existing full
security exit.  No entry, rank, funding, account risk or execution rule changed.

Tests were written first and failed because the implementation module was
absent.  After the minimum implementation, focused tests passed 4/4, ordinary
tests 87/87, and research tests 414 passed with 9 skips.  The disabled control
is frame/order-identical to the unchanged two-position price book.  Actual
additions update weighted basis; targets and unfilled requests do not; full
liquidation clears campaign state.

## Fixed paired screen

| scope | control wealth | treatment wealth | wealth change | control MDD | treatment MDD | event count |
|---|---:|---:|---:|---:|---:|---:|
| union | 4.744148x | 1.948528x | -58.93% | 30.56% | 30.46% | 17 |
| common-five | 7.904183x | 7.185036x | -9.10% | 39.14% | 44.46% | 8 |
| leader removal | 1.154679x | 0.838853x | -27.35% | 35.33% | 48.29% | 20 |

The static control-campaign attribution had seen only 6/1/7 events. In the
executable paired account, an early sale released cash, changed subsequent
admission and basis paths, and expanded the event chain to 17/8/20. Gross
turnover rose from 39.03 to 44.31 in union, 36.22 to 39.43 in common-five, and
47.48 to 55.62 in removal. The treatment therefore re-created the repeated
switching problem it was meant to remove. A locally pure losing-cycle marker is
not a valid portfolio intervention when its released cash is recycled.

Decision: `REJECTED_PAIRED_SCREEN`.  No 2026 or final acceptance matrix was
run.  Do not tune EMA span, confirmation length, basis margin, exit fraction,
cooldown or released-cash handling, and do not reuse the small union MDD
improvement.  The exact screen was deterministic: repeated compact output
SHA256 `dcb7fa47fc4280ed7201fe7f9c464549c6224640f0b299ab57ceae226b896020`.

Source SHA256:

- implementation: `d956de157f9fa4d869570a8424f6121b6143ab21aac89e81f71cedae519d6892`
- tests: `ae1cc69fc19365861e09c85f528396d46b82f1d6ac78728d6445a823e5ff6b8d`
- fixed contract: `1e301d324db7c7a2993bcbf38a0dddf168862ab799274f8bc286e42a2ee6bf2a`
- runner: `29f02cc3770825e612c1f06fc9fb7114b9948927fadfc75497c06579252744f4`

