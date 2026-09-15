# Alpha-first batch: profitable-trend shield rejection and reference-account gap

Status date: 2026-09-15 UTC

## Immutable identities

- Original PR #1 remains open, Draft and unmerged.
- Live main at recovery: `966f7428658431c9da5a6fcaf7d9b9d5c446f2cd`.
- PR head: `efffb9a428258e921feaff125011cba46223bb18`.
- Measured research source: `f02c17dbf57b02f5cac48b88e5b46ac45a7e2b30`.
- Measured source tree: `0b52c21ea3ee6239ed50a2c616e764bdf63c6b54`.
- Hosted evidence run: `34995318510`, completed successfully.
- Hosted correctness run: `34995318623`, completed successfully.
- Hosted evidence commit before this note: `8188c94221a66e6de06a19a4ad6a5e007936a1da`, tree `ef8abe282b63b1eeb7827fe6e9491922b6b0b234`.
- Complete hosted payload: `nonlinear/f02c17dbf57b02f5cac48b88e5b46ac45a7e2b30/34995318510/`.

The first source `e8ba999758178e2e74167972204904c14c78bd82` failed while serializing trace-only NaN values from unrelated unheld symbols. That failure is retained at evidence commit `cee06df17422b3292226f8ede494d4b9948cf8c1`. A regression test reproduced the failure; the trace was narrowed to shielded positions, whose eligibility predicates guarantee finite values. The fixed source passed 87/87 ordinary tests and 363 research tests with 9 skips.

## Fixed paired alpha screen

The same pre-2026 candidate was evaluated on all three required core scopes. No 2026 treatment or final matrix was run after rejection.

| scope | control wealth | treatment wealth | relative wealth | control MDD | treatment MDD | orders control/treatment |
|---|---:|---:|---:|---:|---:|---:|
| union | 4.2311864967x | 3.5202595782x | -16.8021% | 29.6451% | 36.8051% | 149 / 140 |
| common-five | 5.7180461463x | 6.1328140800x | +7.2537% | 31.5516% | 31.5516% | 118 / 114 |
| joint optical-leader removal | 2.3582051229x | 2.4106330596x | +2.2232% | 33.0300% | 32.8387% | 161 / 146 |

Hosted classification is `REJECTED_PAIRED_SCREEN`: `wealth_nonregression_all_scopes=false`, `strict_wealth_improvement=true`, `advance=false`. The implementation and failure evidence remain immutable; do not relabel this candidate, search nearby EMA/return windows, or run its 2026/final matrix.

The earliest material union divergence occurs after retaining `sz300308` through the 2023-06-21 market shock. Its control campaign exited 2023-06-26 at approximately +62.6% on buys; the shielded path continued to 2023-07-06 and retained only about +49.8%. Later shielded `sh688256` campaigns similarly gave back return. In 2025 `sh601869` kept a strong percentage return but had a much smaller capital base because earlier losses had already compounded. Therefore “profitable + positive 20/60 trend” cannot distinguish a normal correction from a mature parabolic exhaustion event. This is an economic falsification, not a request for a tighter filter.

## Actual-account alpha gap

This diagnosis used only authenticated transaction/equity outputs already preserved for the corrected WorkBuddy, Trae and Dumate common-five accounts plus the independent quant account. No private reference source was read or copied and no reference portfolio was rerun. Reference campaign PnL below is gross from recorded trade prices, while quant campaign PnL is net of recorded fees; terminal wealth remains the authoritative normalized comparison.

Full-history closed-campaign comparison, 2023-01-03 through 2026-09-11:

| account | closed campaigns | win rate | median campaign return | median hold | mean hold | loss PnL |
|---|---:|---:|---:|---:|---:|---:|
| corrected WorkBuddy | 39 | 76.92% | +11.60% | 55 days | 60.4 days | -0.690m CNY |
| quant main | 46 | 50.00% | -0.33% | 21.5 days | 30.9 days | -3.907m CNY |
| Trae corrected-fee diagnostic | 177 | 44.07% | -0.89% | 7 days | 10.6 days | -12.235m CNY |
| Dumate corrected-next-open diagnostic | 65 | 29.23% | -5.55% | 39 days | 65.6 days | -0.629m CNY |

WorkBuddy recorded gross campaign PnL is about 39.789m CNY versus quant net campaign PnL about 19.650m CNY. Of the approximately 20.139m descriptive gap, about 16.922m comes from larger positive campaigns and about 3.217m from smaller losses. The dominant missing capability is therefore profitable campaign discovery/participation and lifecycle, not another defensive patch.

WorkBuddy-minus-quant campaign PnL gaps by common-five symbol are approximately:

- `sz300394`: +6.058m CNY.
- `sh603986`: +5.973m CNY.
- `sz300502`: +4.449m CNY.
- `sz300308`: +2.866m CNY.
- `sh688008`: +0.794m CNY.

High-information lifecycle witnesses:

- `sz300394`: WorkBuddy entered 2025-12-05 and exited 2026-03-10 for about +70.4%; quant entered 63 calendar days later and captured about +26.5%. This is principally opportunity discovery/late entry.
- `sz300308`: WorkBuddy entered 2025-06-16 and exited 2025-09-05 for about +226.8%; quant entered 29 days later and exited the same day for about +140.5%. This is entry timing, not exit.
- `sz300502`: the 2025 campaign timing and return were similar, showing that capital-base compounding matters after earlier missed campaigns.
- `sh603986`: in the 2025-09 campaign quant exited 49 days earlier and captured about +2.4% versus about +16.0%; this is premature exit.
- `sz300502`: WorkBuddy held a profitable 2026-04-09 through 2026-07-14 campaign for which quant had no overlapping campaign; this is a complete missed/re-entry opportunity.
- `sh603986`: in 2026 quant actually entered earlier and earned a higher percentage return than WorkBuddy, but deployed far less capital because of its weaker prior compounded account path. Do not misclassify this witness as an entry rule failure.

The existing independently written coherent account is the strongest reusable evidence: full-history common-five wealth is `22.7716733036x` and joint leader-removal wealth is `2.9915298718x`, but union wealth is only `1.7481913470x` versus incumbent `5.6247932765x`. Thus one existing lifecycle can exceed the strongest common-five reference and improve removal, while broad-universe distractor admission destroys the union account. Do not splice per-pool winners; the next architecture must use one causal rule on every supplied universe.

## Next distinct direction

Investigate one unified strong-theme/co-leader campaign architecture, not a parameter neighbor:

- Economic hypothesis: coherent-style lifecycle is capable of sufficient alpha once it owns a genuine theme cohort; its broad-universe failure comes from treating isolated positive ranks as equivalent to synchronized sector/theme leadership.
- Expected source of improvement: earlier participation in simultaneous leaders, persistent ownership through the shared theme campaign, fewer isolated union distractors, and re-entry while the theme remains causally active.
- Structural boundary: form theme state only from supplied-universe, causal price/volume/breadth and existing generic sector metadata. No reference trades, source code, ticker/date keys, per-pool switches, leverage, shorting, or changed execution/risk/cost assumptions may enter runtime decisions.
- Minimum validation: preregister one fixed candidate and compare it with one unchanged control through 2025-12-31 on union, common-five and joint leader-removal. Record wealth, MDD, turnover, exposure, lifecycle events and exact ledger reconciliation.
- Failure rule: reject without neighboring thresholds if any core scope loses wealth, or if the architecture cannot materially repair union while retaining the common-five/removal economic advantage. Do not run retrospective 2026 or the final matrix after a failed core screen.

Writer state is released. No unpushed local source is owned by this batch. Reread live refs and Actions before taking over. Original economic acceptance remains `NOT_MET`; PR #1 must remain Draft/unmerged and the continuation must remain active.
