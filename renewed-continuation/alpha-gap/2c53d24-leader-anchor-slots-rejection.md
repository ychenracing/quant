# Leader-anchor marginal slot: fixed screen rejected

Date: 2026-09-15 UTC
Registration: `fbab56360b555bc69c9a771b4b27c5db1d603c61`
Measured source: `2c53d24a0fbc1c44fcccf16ab2634434258b41ca`
Source tree: `94ccd0a07f32783178ec4b860ef0e6cde9073a8e`
Economic run: [35006251924](https://github.com/ychenracing/quant/actions/runs/35006251924)
Correctness run: [35006251956](https://github.com/ychenracing/quant/actions/runs/35006251956)

## Fixed contract

The treatment preserved the unchanged parent's highest-ranked currently
admissible individual as slot-one anchor. The marginal second slot preferred
the highest-ranked admissible generic-sector peer and otherwise fell back to
the unchanged parent second candidate. It changed no score, capital weight,
numeric threshold, risk rule, data, universe, seed, cost, liquidity or
execution rule. Existing holdings were not sold merely because of the mask.

Five targeted tests failed before the implementation existed and passed after
implementation. The complete local verification was 87/87 ordinary tests and
381 research tests passing with 9 skipped. Hosted correctness also passed.

## Hosted fixed-pair result

| Scope | Control wealth | Treatment wealth | Relative change | Control MDD | Treatment MDD |
|---|---:|---:|---:|---:|---:|
| union | 4.7441484288x | 3.0374476183x | -35.97% | 30.5579% | 33.8044% |
| common-five | 7.9041830487x | 8.6435865645x | +9.35% | 39.1356% | 39.7191% |
| joint optical-leader removal | 1.1546791489x | 1.3319359407x | +15.35% | 35.3310% | 31.0278% |

Decision: `REJECTED_PAIRED_SCREEN`.

The treatment improved wealth in two narrower scopes and materially improved
leader-removal MDD, but failed the preregistered all-scope non-regression and
strict-union requirements. No treatment 2026 or final-acceptance run was made.

Hosted archive SHA-256:
`b1d41a619a672a8d005231bd3f9e00ab8770abb768ed19cb9f751755f378f4c9`

Selection SHA-256:
`15c4d7bc71b05fc293e5258b7b6ada60684cb19c4f7fcec652a4d34b82202ecd`

## Path diagnosis

The leader anchor fixed the prior candidate's complete rejection of isolated
leaders, but marginal-slot admission remained strongly path dependent in the
34-name union. Most of the relative loss formed in 2023 Q2: the
treatment/control wealth ratio moved from 1.043 at 2023-03-31 to 0.635 at
2023-06-30.

On 2023-05-18 and 2023-05-19 the causal treatment state correctly named
`sz300308` as anchor and `sz300502` as sector companion. The control entered
both from the 2023-05-19 signal and realized about +1.058m and +0.272m.
Treatment did not enter them at that point; earlier ownership choices left
legacy inventory/obligations in the book and it entered a later
`sz300308`/`sz300394` pair only from 2023-06-13. Aggregate symbol campaign
PnL versus control was lower by about 1.705m on `sz300502` and 1.446m on
`sz300308`.

This is retrospective path diagnosis, not acceptance evidence. It shows the
remaining union failure is not primarily theme discovery at the signal close:
the correct leaders can be observed while stale funded inventory prevents a
timely ownership handover.

## Closed direction and continuation constraint

Do not tune the companion rank, sector preference, fallback rule or nearby
parameters, and do not repackage this candidate. Theme evidence should not
continue as an admission mask.

A distinct next hypothesis may test a causal campaign **handover**: when the
unchanged parent already observes a complete two-name same-sector leader
cohort, explicitly release funded holdings outside that cohort so actual fills
can free capacity for the next-session campaign. It must leave ordinary
admission unrestricted, preserve all actual-share/T+1/cash/liquidity
obligations, use no ticker/date/pool branch, and be preregistered as a fresh
fixed pair. The first validation question is whether handover timing, rather
than another selection gate, restores the 2023 Q2 union campaign without
regressing common-five or removal.
