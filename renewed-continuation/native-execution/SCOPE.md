# Native inventory correctness reconciliation

Continue original quant PR #1. This is an execution-correctness comparison, not a new economic strategy hypothesis. The closed three-hypothesis and renewed two-hypothesis budgets remain closed; no candidate grid, eligibility change, new ranking rule, or acceptance change is authorized by this record.

## Verified recovery

- Original PR: OPEN, Draft, unmerged; research/independent-tech@4fa306fdcde2f9a77d941200ef7ff2f988c045e5.
- Live main: bbc002449093e458868a8748687df023958ba283; PR metadata's older base SHA is not used.
- Research source recovered from run34926804636/artifact10380089653: 32304314270586d078d527aa2860aab78a09f613, exact reconstructed tree253804819b1d43c17c066eb1e353f40c3eaa762a. ZIP f7a19e2fcb152c4aaa8ea4e68ffd58e193b0477fc52d9ec30c85733db25a4883; tar316471d9b036f3c4e6cfc0899a1e59858fcd8c353db8d52267f062da838170be; all63 manifest hashes match. No research accounts were rerun during recovery.
- Frozen inputs recovered from main run34794779702/artifact10329173115, tar6f7864e8f60001790806dcedb2fb7abc5960fb992c936ee9ea2be02069f3f353; input fingerprints must be checked before execution.
- Read-only private reference recovered through the authorized GitHub artifact channel: trades run34135770719/artifact10023909983, ZIP514c79ece6f74d56e0a461376d52a490103d58883fbdef233fdbb372a6552064. Workbuddy implementation matches the frozen Git blob e19e611cd332e7a01c0f8677e0ae60600b295c6d. Private source stays outside the quant source tree and is not published.
- No local prior checkout/process or running/queued quant Actions was found. Latest PR comment5674674697 records the preceding worker's completed bounded review. This does not claim visibility into unmounted workers; recheck before shared writes.

## Fixed task

Reconcile the previously recorded workbuddy/common-five same-session sale of newly bought inventory on 2025-04-28. Recover or reproduce the native control using its unchanged settings, frozen data, input adapter and reference source. Require its raw trades hash a9620cef60d7f18623af3d676ac7aaaa29d7de9613712812764998eecb406adb and reported wealth20.221072996623576, or explicitly retain any runtime mismatch without treating it as exact equivalence.

Implement the minimum isolated comparator correction needed to restrict actual sales to beginning-of-session inventory less prior sales. Preserve earlier inventory's sellability after same-day additions, cash/fees and unsold positions. Preserve and retry blocked protective quantities on a later available session rather than deleting obligations. New same-day purchases must not replenish sellable inventory. Keep terminal locked inventory marked to market instead of inventing a same-day liquidation. Do not alter selection, allocation parameters, stop thresholds, original metrics, frozen data or original reports. A reference execution adapter is not a production strategy base.

Validate the original failing case, old-inventory sales after top-ups, repeated sale attempts, partial fills, next-session release/retry, cash reconciliation and unchanged paths. Then measure this one fixed execution correction on the same full common-five account; do not fan out to new strategy variants. Preserve control and corrected curves, fills, settings, identities, tests, execution events and audit findings separately.

Inventory-only correction does not by itself normalize intraday information timing, fee/liquidity semantics, risk-state resets, corporate actions or all four references. Report those limits explicitly; do not label a T+1-only result as fully executable/normalized economic acceptance. Production acceptance remains NOT_MET and original PR #1 must not be merged on this comparison alone.
