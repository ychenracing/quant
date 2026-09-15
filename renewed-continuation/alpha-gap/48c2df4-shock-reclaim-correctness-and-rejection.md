# Shock-reclaim correctness repair and economic rejection

Updated 2026-09-15 UTC.

## Scope and preserved identities

This note closes the prospectively registered `shock_reclaim` candidate without changing its economic hypothesis, thresholds, frozen data, control, scopes or rejection rule.

- Registration: `39b7e7aa3ec63f9ba29d2cfd034facd9c35659bd`
- First requested implementation: `9c61409630e5b8d5911ba4bc0016160bb2135237`
- Corrected source/request: `48c2df4ab101601d09d9feaca82d259452b37240`
- Corrected source tree: `d7e84fa78fc642182b69c0cd9d28066d87e1ab4e`
- Corrected hosted economic run: `35002117259`
- Corrected hosted correctness run: `35002117215`
- Corrected evidence branch commit read before this note: `43a87064fb54c353f453204a2605870c27174025`
- Corrected evidence tree read before this note: `4386f87a6e46f43542e7415641cbf7beaf7988c7`
- Evidence archive SHA256: `b808b22a65c3e9f36dfd70c93bba5a0806d5c114ab4b70f69228f8028717bea6`
- Selection SHA256: `8d6c590b272d5e714539f96efc70410370d4cb80cd57230d70e5f7060232fe03`
- Frozen pre-2026 data identity: `894230b20361fb826d58b27987e87146a090b9e894e7abf091121486f5293157`

The original hosted run `35001162092` is retained. It reported zero reclaim events and equal accounts, but the zero-event trace was caused by an implementation-order defect rather than by absence of eligible historical episodes.

## Correctness root cause and repair

The wrapper observed an actual shock liquidation reaching zero inventory and marked the reclaim record flat. Before the parent policy could observe the same fill and arm its ordinary readmission veto, the wrapper interpreted `flat && !parent.readmit` as a natural release and erased the just-created record. Therefore every newly completed liquidation lost its reclaim memory in the same close.

The corrected source retains a just-completed liquidation through that fill-reconciliation close. A targeted regression test failed before the repair and passed after it. Independent full verification on the exact source completed:

- ordinary tests: 87/87 passed;
- research tests: 371 run, 371 passed subject to 9 existing conditional skips;
- compile and diff checks passed;
- hosted correctness run `35002117215`: success.

## Corrected fixed paired screen

The corrected candidate now emits real lifecycle activity:

| scope | release events | reclaim requests | control wealth | treatment wealth | control/treatment MDD |
|---|---:|---:|---:|---:|---:|
| union | 28 | 4 | 4.231186496672099x | 4.231186496672099x | 29.6451068% / 29.6451068% |
| common-five | 16 | 2 | 5.718046146255741x | 5.718046146255741x | 31.5516350% / 31.5516350% |
| joint optical-leader removal | 13 | 2 | 2.358205122900610x | 2.358205122900610x | 33.0299976% / 33.0299976% |

All six treatment/control accounts are exactly equal in wealth, MDD, orders, turnover, fees, slippage, exposure and ledger outcomes. The eight reclaim requests occur when the unchanged parent policy's ordinary readmission clock matures on the same close, so the candidate does not create an earlier executable opportunity.

Hosted decision: `REJECTED_PAIRED_SCREEN`; `wealth_nonregression_all_scopes=true`, `strict_wealth_improvement=false`, `advance=false`, treatment 2026 runs = 0. This is a correctness-fixed rejection, not a new candidate or a PASS. Do not tune rank, reclaim price, or waiting-time neighbours and do not run this treatment over the 2026/final matrix.

## Alpha implication and next distinct direction

This result closes selective shock re-entry as an independent alpha source under the current parent lifecycle: recovery eligibility and the parent's natural readmission clock coincide. The next candidate must change opportunity formation rather than add another release wrapper.

Proceed with one unified causal strong-theme/co-leader campaign state across union, common-five and joint-removal:

- form and persist a synchronized theme cohort using only causal supplied-universe price/volume/breadth and generic sector metadata;
- admit capital only after cohort activation;
- use the same rule and lifecycle across all scopes, with no ticker/date keys, pool-specific branches, reference-runtime inputs, parameter grid, ranking-weight tweak or risk patch;
- preserve the coherent lifecycle advantage on common-five/removal while materially repairing union distractor admission;
- preregister one fixed control/treatment pair and reject without neighbours if any scope loses wealth or union is not materially repaired.

PR #1 remains Draft/unmerged and economic acceptance remains NOT_MET.
