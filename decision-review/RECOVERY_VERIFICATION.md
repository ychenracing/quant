# Independent recovery verification — original quant PR #1

2026-09-15. This supplements `REVIEW.md` and the existing cumulative `budget.json`; it does not replace their decisions or introduce a fourth hypothesis. The three registered treatments are rejected. Finite economic research is complete; original economic acceptance remains NOT_MET and main is not authorized for promotion.

## Checks completed without new portfolio runs

The seven supplied/downloaded archives passed all 10,340 top-level manifest member hashes: main 6,048; fixed 3,589; parent 149; admission 167; protected 156; readmission 141; healthy-hold funding 90. The restored source tree is exactly `2b0e1b0e3911377c0f3232971c4e96ac007a9cd8`, belonging to `07f4f32f2a856866bb774e128470b437ceb7f779`. Its 11 production file hashes and 19 recorded research dependency hashes match the archived plan. This does not replace research source identity with a production-package hash.

`reconcile.py` performs arithmetic on recorded fills only. It does not call a policy selector, change an economic rule, or rematch orders. It verified 15 preserved accounts: nine full-period accounts from parent source `4d0c9ec1ada94c4e23f85a329b880adbe0bd5aa1`, run `34910777906`, and six pre-2026 paired accounts from `07f4f32f`, run `34919407590`.

Cash reconciled exactly to every recorded close. Minimum reconstructed cash after a fill across these accounts was CNY 15.592021975433454; no account borrowed to fund a recorded buy. Every recorded signal preceded its fill, and daily sales remained within opening economic-unit inventory. Maximum daily PnL reconciliation error was CNY 1.4901161193847656e-08. These checks validate the recorded adjusted-unit proxy, not actual shares, corporate actions, tax accounting, or private reference execution.

The existing whole-account vectors for main/incumbent, retained control and readmission remain mutually non-dominated when the three pools' wealth, drawdown and filled-order counts are considered together. This is a descriptive result for that recovered set, not a new selector or a claim of global optimality. Original source identities are retained in each audit account; no old run is attributed to current HEAD.

## Confirmed economic and execution observations

The final fixed pair increased pre-2026 wealth by 25.481348%, 10.643738%, and 1.519659% in union/common-five/joint-leader-removal, but worsened drawdown by 2.641888, 4.477697, and 3.997662 percentage points. The immutable pair screen therefore rejects it. These are not full-period treatment results.

| Retained control, actual recorded return | 2026-06-22 through 2026-08-31 | 2026-07-01 through 2026-08-31 |
|---|---:|---:|
| Union | -13.230720% | -12.207740% |
| Common five | +0.238208% | -13.831656% |
| Joint leader removal | -6.464435% | -13.970407% |

All three first reduced total risk budget in these windows on the July 2 close, with associated actual sells on July 3. July 2 actual exposures were 72.134907%, 94.698158%, and 58.576638%, respectively. The union/removal budget reductions on July 17 had no associated sell fills; the exposures were already below the new cap, so a lower target alone is not an unfilled sell. All three were flat on August 31 and remained flat through the frozen September 11 data end.

There were 13 blocked attempts across the three full-period retained controls, including 11 sells carrying a PROTECTIVE_INVENTORY_PENDING signal reason. None occurred inside either 2026 crash window. Every such blocked sell has a later actual sell for that symbol in the ledger. `next_actual_sell` records that chronology only: it does not prove the later order discharged an identical original quantity obligation. Blocked buys' next sell dates are likewise not evidence that the blocked buy filled. No blocked attempt is counted as a completed protective exit.

## Neighborhood coverage: later implementation is not a completed measurement

The old fixed archive's ten unsupported parameter-neighborhood rows are not the whole history. A later implementation attempted the check at source `06ad8aa018cb0890c6b0b60e1dae3de24535f8c5`, run `34881928214`. Its preserved `fixed-validation/.../failure.json` shows the default-equivalence guard failed on joint leader removal, 2023-09-01, before economic measurement. The source-bound failure is retained; no tolerance was widened and no new neighborhood run was launched in this audit. Neither implementation existence nor a completed workflow proves those economic cases passed.

## Reproducibility and publication integrity

Run the saved `reconcile.py` with the recovered evidence root, exact source snapshot and an output filename. The local runtime was Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, Linux 6.18.44/glibc 2.41. Original hosted runtime identities remain in their own archives; this local arithmetic verification does not relabel hosted generation.

- Script SHA256: `96a87fafb66b61828071d2aa1338ea2f83c0fcd23a3dff0317b9eb986a59a832`; verified remote Git blob `494e2093db6b17245eb54eeee53ab949e23770e7`.
- Original pretty-printed audit output SHA256: `d541618396a80ffc84783f32392950ae4fe96c850e8f09e963cbca9dc5e7f31b`. `recovery-audit.json` preserves every field/value with compact JSON formatting; expected remote Git blob `3f00b26fba88e38607db77983b768b9037e52835`.
- An initial local invocation omitted the existing catalog sector map, so its fingerprint guard stopped before account reconciliation. The invocation was corrected to match `decision_review.main`; no data, expected hash or strategy was changed.
- Initial transport commit `3382b284751b53b64429795b748204be182c0994` contains a checksum-invalid compressed audit bundle, blob `8cdd8786b354c6f6ac73187a28039778253b596e`. It is INVALID, not canonical evidence. History is retained. Commit `a607bd2017050b7ac0f0a4168d3db4a0b28b981f` replaces it with the complete readable audit output; the script is separately preserved at `41a269a561ee70f23d831e55bf716f479e7f290a`. The original seven research archives were never changed.

No new historical portfolio simulations, new economic hypotheses, private reference code publication, force-push, or main changes were performed by this verification. Full original economic acceptance is still not established. Remaining substantive needs are execution-equivalent reference evidence and genuinely new, causally usable information or an explicitly authorized change of return/risk trade-off—not another unbudgeted neighboring patch.
