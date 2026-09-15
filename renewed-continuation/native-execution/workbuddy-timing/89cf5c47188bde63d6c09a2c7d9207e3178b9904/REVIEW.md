# WorkBuddy common-five execution-timing correctness result

This closes the fixed correctness comparison registered in `renewed-continuation/native-execution/workbuddy-timing/SCOPE.md`. It is not a new quant strategy candidate, does not consume or reset either closed economic research budget, and does not authorize PR #1 promotion.

## Fixed source and comparison

Quant research adapter source: `89cf5c47188bde63d6c09a2c7d9207e3178b9904`, tree `b930077f108423039b9e29197278d0fb38efee52`, parent `6c25731bfbad5149c0f8144423883edb6247ef76`. The private WorkBuddy reference remains external and hash-pinned to Git blob `e19e611cd332e7a01c0f8677e0ae60600b295c6d` / SHA256 `d74e8cd6c9a68823a08c5ed5526d72e65dce7b9cdc7568eeb38d1f9d80483a29` under frozen `ychenracing/trades@5575d1b1b79fab405b92cfb7d164c7054a8bd826`.

The comparator keeps the already verified T+1/sellable-inventory correction and changes only two timing rules that were noncausal under the after-close/next-session contract:

- intraday stop thresholds use the latest fully completed prior session ATR and close rather than the same session's completed high/low/close-derived ATR;
- a hard drawdown breaker established by the completed close schedules liquidation for the next available open rather than receiving that same close as an executable fill. The native open-derived hard breaker remains unchanged.

Selection, eligibility, allocation, stop multipliers, drawdown thresholds, risk gates, fees, slippage, frozen prices, universe and original native reports are unchanged. `settings.json` is byte-identical to the prior inventory-only corrected account.

## Measured result

Common-five, CNY2,000,000 initial cash, 896 sessions from 2023-01-03 through 2026-09-11:

| account | wealth incl. principal | full MDD | fill rows | T+1 violations |
|---|---:|---:|---:|---:|
| original native | 20.2210729966x | 25.331952% | 107 | 1 |
| inventory-only corrected | 20.8197253821x | 25.295342% | 109 | 0 |
| timing + inventory corrected | 17.8279457167x | 23.670313% | 108 | 0 |

Relative to the original native headline, the fixed timing+inventory account is 11.8348% lower in terminal wealth and 1.6616 percentage points lower in MDD. Relative to the inventory-only correction, terminal wealth is 14.3699% lower and MDD is 1.6250 percentage points lower. This is a material execution-timing sensitivity in the strongest historical comparator, not a quant production improvement.

The first equity divergence versus the inventory-only account is 2023-08-08. Its close-derived hard-breaker sell appears on the next session, 2023-08-09, rather than receiving the 2023-08-08 close. Additional deferred native flat-reset completions occur on 2024-11-18 and 2025-04-29. No outstanding protection or terminal position remains.

Requested decline windows under this corrected diagnostic are:

- 2026-06-22..2026-08-31: return -12.059847%, local MDD 14.326132%.
- 2026-07-01..2026-08-31: return -14.326132%, local MDD 14.326132%.

Full-period MDD remains 23.670313%, still above the original 18% risk ceiling. Therefore this correction does not make the reference itself satisfy every original risk requirement and cannot be used as an acceptance waiver.

## Verification

- 26 targeted inventory/timing tests passed against the exact private reference, including an explicit RED witness that the prior inventory-only transform retains same-session ATR and same-close hard-breaker behavior.
- 87 ordinary offline tests passed locally.
- exact source commit `89cf5c4` Offline correctness run `34936260906` completed successfully; this CI does not execute the private reference and is not economic acceptance.
- the measured account is bound to producer commit `89cf5c4`; the full 34-name/896-session frozen input fingerprint remains `d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b`.
- independent 896-session cash/position/NAV replay has zero inventory violations, zero holding mismatches, zero recorded-cash mismatch, and maximum NAV absolute error `7.450580596923828e-09`.
- actual inventory regression passes on the final measured account; final holdings and outstanding protection are empty.
- corrected economic file hashes are retained in the payload. The private reference source is not included.

## Interpretation and remaining scope

The earlier inventory bug did not inflate WorkBuddy; fixing inventory increased its headline. In contrast, enforcing two independently identified causal timing constraints materially reduces the strongest native common-five headline. This improves the credibility of the target-distance analysis but does **not** silently replace the original native benchmark or lower the user's economic target. Native and correctness-adjusted results remain separate evidence classes until a complete four-reference normalized execution comparison exists.

Other WorkBuddy semantics such as the current-session ATR availability check at the open, liquidity diagnostics, corporate-action/actual-share accounting and any nonbinding same-close guards remain to be classified. The currently inspected common-five profile has `cap_enforce=false`, `vol_stop_k=0`, `mkt_enforce=false`, `vol_reduce=0`, `overbought=0`, so several theoretical look-ahead inputs are nonbinding in this account; this is not proof for every pool.

Next correctness work should use the already authenticated raw streams to classify material execution differences in ChatGPT, Trae and Dumate without blindly rerunning historical cases. No additional quant economic mechanism is authorized by this result. Original quant economic acceptance remains `NOT_MET`, production delivery remains `NOT_COMPLETE`, and original PR #1 remains Draft/unmerged.
