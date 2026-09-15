# Source-bound inventory correction evidence

## Scope and current status

This records the completed fixed comparator correction at research/expectation-learning@6c25731bfbad5149c0f8144423883edb6247ef76, tree2fcd1232dee08930372c8aedfc7b3758f05bdeb2, parent32304314270586d078d527aa2860aab78a09f613. Only research/native_inventory.py and research/test_native_inventory.py were added. Original PR #1 head and main were not modified. No new trading hypothesis, closed-budget reset, parameter search, economic matrix or merge was performed. Original economic acceptance remains NOT_MET and production delivery NOT_COMPLETE.

The public adapter instruments the exact private reference in memory; the private original and a corrected private source file are not published. Reference Git blob e19e611cd332e7a01c0f8677e0ae60600b295c6d is enforced along with all21 structural anchors. Both the original policies and this adapter's limitations remain explicit in the corrected identity file.

## Measured once at this source

Common-five/profile-b,2000000 initial cash,896 sessions,2023-01-03 through2026-09-11. The original reference, frozen market/settings and input adapter are unchanged. Full34-name input fingerprint d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b.

| Metric | Original native | Inventory-only corrected |
|---|---:|---:|
| Wealth including principal |20.221072996623576|20.81972538210324|
| MDD fraction |0.25331952059019447|0.2529534216033119|
| Fill rows |107|109|
| T+1 violations |1|0|
| Final cash |40442145.99324715|41639450.76420648|

Relative terminal wealth increases2.9605371860%; MDD decreases0.0366098987 percentage points. This is a corrected reference result, NOT a quant production improvement. The same-day inventory violation cannot explain away the reference's higher headline or justify reducing the original goal.

The actual violating path was the2025-04-28 close hard-drawdown liquidation: sz300502 bought52100 units that session, then sold52100 despite zero opening inventory. The corrected adapter retains those units and quantity-bound protection, retries at the native next-open price on2025-04-29, and delays the native flat reset until actual clearance. Three inventory events are recorded: block, discharge, deferred reset. Outstanding protection and terminal positions are empty. Earlier NAVs are identical; first NAV divergence is2025-04-28.

## Verification and identities

-21 inventory/private-helper tests PASS, including4 integration tests against the hash-pinned private reference;2 reference-input identity tests PASS.
-Actual full-account regression: original RED with exactly52100/0 violation; corrected GREEN. Both logs retained.
-Neutral AST fixture: equity.csv,native_frame.csv,trades.csv,settings.json all byte-identical to original. This was a correctness fixture, not a new trading hypothesis.
-Independent896-session fill/cash/holding/NAV reconciliation: maximum NAV absolute error7.450580596923828e-09; corrected trace cash error0.0; no holding mismatch; no outstanding protection. Full reconstructed daily ledgers retained.
-Exact source6c25731 Offline correctness run34931912796 completed successfully. It does not execute the private reference. Source-verification artifact10381209387 digestb0cc543c5312a361bb30f45043a9847ddecc0f3625959928a4a4990fedd62c83 was read as GitHub metadata, not downloaded/re-extracted in this continuation.
-Original trades SHA256 a9620cef60d7f18623af3d676ac7aaaa29d7de9613712812764998eecb406adb matches the earlier frozen anchor.
-Corrected trades SHA256 fa9dad34ad7fab4b4531927668a404741f8bf623c2d29f9dff97368010014d3e; corrected equity4b81812b23e6fc52f8a07fa1ff6eb8f28226827729cbec295518cb2ace8ee6da.
-Adapter SHA256 f0611e4536853adc3197fb0a5d354d66f11494d7280a1bf19d7024bf5dbcdc3f; tests1e49503dd938a2b631b64a86402fb3504bb1186d448e41248dd2572dd27148e9.

## Durable payload and concurrency reconciliation

transport.json binds four raw binary chunks of one35520-byte XZ tar, SHA2569f8dc138c6af0729606456af3aab13fa3ef056008a11898c60f0f417801f13f5. It contains42 members:41 actual files plus MANIFEST.json. All41 content hashes were checked against extracted archive bytes. Contents include full original,neutral and corrected outputs, corrected inventory trace, independently reconstructed ledgers, RED/GREEN/run/test logs, verifier/fixture scripts, receipt.json and RECOVERY.md. No private reference source is included.

During final publication another continuation advanced research/evidence to5a5a666d675dd88d0098726cfa2471900e716aea and preserved a separately identified private correction at d9c6b3bc0f089ea482336bfa1f6c1b86d9934a9b. Its reported wealth/MDD/fill count agree with this measured account, but its private-source hash and raw archive differ. We preserve both identities rather than claiming bytewise equivalence or rewriting one provenance as the other. No further account was run after observing that overlap. This source-6c25731 directory is additive; its full source, generic tests, neutral-equality evidence and daily accounting traces are independently recoverable. Neither previous archive nor other worker's reports, progress ledger or comments are overwritten.

## Remaining work

This closes only the already identified workbuddy/common-five inventory issue. It does not normalize same-day ATR/close information timing, fee/liquidity/price availability, corporate-action or actual-share accounting, native risk resets, or the other three references. The corrected MDD is still above the original18% ceiling. Continue authenticating the four-reference source/input/output identities and execution-equivalence matrix without repeating this measurement or reopening closed strategy budgets. All original joint quant acceptance remains required before PR #1 can merge.
