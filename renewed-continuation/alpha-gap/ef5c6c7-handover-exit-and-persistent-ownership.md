# Alpha gap: handover, first-exit audit and persistent ownership

Status: diagnostic handover rejected; `persistent_leader_ownership` rejected by its fixed paired screen. Original economic acceptance remains `NOT_MET`.

## Bound identities

- Frozen full market SHA-256: `d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b`.
- Pre-2026 selection market SHA-256: `894230b20361fb826d58b27987e87146a090b9e894e7abf091121486f5293157`.
- Parent source and hosted-control identity: `2c53d24a0fbc1c44fcccf16ab2634434258b41ca`; local reconstructed controls matched hosted equity fields within `1.87e-9`.
- Persistent-ownership preregistration: `fc329427c50c29e994f622330fbaa6d626f0d50b`.
- Implementation: `0970e07a6403c868e20deef02d85b6956d8f4906`.
- Tests: `7f9b09e9c08462b99d4043e16ed2cd00b1a1f1e6`.
- Result record: `ef5c6c78ce86d4ee38a1a1c49e004a445b4727cb`.
- Source hashes: implementation `fd7549a02d895042645f277d8faaaa64372f56d6f63ecadb22341b45fba40730`; tests `0bf6c8e9f768c7b3682b4ceca8fa03ed1a64e2818e6eed1d588b0ff2b2cee28e`; contract `ad67a74efcf5eddbbe1a658d096fcb50902b70c695968a7200ed99500624f4c0`.

## Campaign-handover diagnosis

A full reconstruction used the unchanged parent `trend_book.Owner(positions=2)`, actual filled holdings and exact original same-sector top-two formation rule. Among episode onsets with an out-of-cohort holding still receiving a positive parent target:

| Scope | actionable onsets | leader 10d mean | blocker 10d mean | mean spread | positive spread |
|---|---:|---:|---:|---:|---:|
| union | 61 | 3.78% | 3.33% | +0.45pp | 32/61 |
| common-five | 46 | 5.22% | 6.58% | -1.36pp | 23/46 |
| leader-removal | 43 | 0.52% | 0.84% | -0.31pp | 21/43 |

The originally suspected 2023 Q2 union handover exists, but the generic causal condition has no robust cross-scope edge. It is diagnostic-only rejected, was not preregistered as an economic candidate, and must not be implemented or retuned.

## First ordinary-exit intent audit

An instrumented parent reproduced the hosted controls within `1.87e-9` and inspected the first close where an actually held name newly hit the unchanged ordinary security-break latch. There were 10/6/9 intents on union/common-five/removal. The proposed causal condition—currently profitable holding, a same-sector peer currently parent-admissible, and no account-risk cut—had exactly **0 activations in every scope**. Therefore the attractive post-exit rallies in episode summaries cannot be captured by this declared rule; no exit-delay candidate was implemented.

## Fixed persistent-ownership screen

The fixed standalone treatment let the unchanged broad market state gate all new units, but made only the unchanged security broken predicate authoritative for disposal of already funded units. It did not tune a shock threshold or revive the rejected selective profit shield.

| Scope | control wealth | treatment wealth | change | control MDD | treatment MDD | orders control/treatment |
|---|---:|---:|---:|---:|---:|---:|
| union | 4.744148 | 3.056690 | -35.57% | 30.56% | 48.60% | 145 / 94 |
| common-five | 7.904183 | 10.904526 | +37.96% | 39.14% | 44.60% | 164 / 76 |
| leader-removal | 1.154679 | 0.843213 | -26.97% | 35.33% | 65.99% | 169 / 96 |

Maximum ledger reconciliation error was `5.48e-9` CNY. Focused tests passed 6/6; ordinary tests passed 87/87; research suite ran 387 tests with 9 skips and no failures.

The candidate is `REJECTED_PAIRED_SCREEN`. It proves that materially longer ownership can unlock large common-five alpha, but unconditional persistence also compounds broad-universe and removal distractors. Do not search hold windows, shock thresholds, partial-book variants or repackage this result.

## Next bounded alpha work

The next high-information step is not another risk wrapper. Attribute the rejected treatment's actual profitable and losing ownership episodes at entry and over their first causally observed lifecycle, using only contemporaneous price/sector/market state. The required question is whether a generic, prefix-causal **campaign quality state** can separate durable leaders from persistent distractors across all three scopes. Only a cross-scope discriminant with adequate activation count may justify one new standalone entry-plus-lifecycle engine. Otherwise stop wrapping the current parent and redesign the alpha source itself.

PR #1 remains Draft/unmerged; production and main are unchanged.
