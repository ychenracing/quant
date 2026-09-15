# Four-reference accounting-basis static audit

This is a correctness-only continuation of original quant PR #1. It creates no quant strategy candidate, consumes no economic-research budget, changes no frozen threshold/data/universe, and publishes no private reference source.

## What is now established

The exact native harness (`32f310e...`) loads the frozen 34-name market once, explicitly records that qfq input is not an actual-share corporate-action ledger, and passes the same adjusted OHLCV/economic-unit frames to all four authenticated private references. That means corporate-action/actual-share accounting is a **shared comparison-basis limitation**, not evidence of an isolated bug that can be "fixed" in one comparator. No synthetic corporate-action return is produced here.

The fee assumptions are not uniform. Quant uses 2.5bp commission, CNY5 minimum commission, 0.1bp transfer assumption, 10bp slippage, and the historical sell-stamp schedule (10bp before 2023-08-28, 5bp thereafter). The authenticated native reference paths use:

| reference | commission | minimum | sell stamp | transfer in inspected path | slippage |
|---|---:|---:|---:|---:|---:|
| ChatGPT | 2.5bp | CNY5 | 5bp constant | none found | 10bp |
| Trae | 2.5bp | CNY5 | 5bp constant | 0.1bp | 5bp |
| Dumate | 3bp | CNY5 | 5bp constant | none found | 10bp |
| WorkBuddy | 2.5bp proportional | no minimum in inspected path | 5bp constant | none found | 10bp |

These differences must stay disclosed as native semantics. They are not all correctness defects: commission, slippage and transfer assumptions may legitimately represent different account/execution assumptions. Replacing them wholesale with quant's costs would change the benchmark definition and can alter cash-constrained trade paths.

## Historical stamp mismatch: measured and bounded enough to stop the fee branch

The preregistered fixed Trae/common-five diagnostic already establishes the one unambiguous historical legal-schedule mismatch. There are 52 sells before 2023-08-28, CNY32,863,762.70 total notional. On the unchanged path the missing 5bp is CNY16,431.88 before compounding/path feedback.

Changing only that sell-stamp schedule changes Trae common-five from 10.674861915267x / 33.107753% MDD to 10.638974540102x / 33.117142% MDD: terminal wealth -0.3362%, MDD +0.00939 percentage points. A second run reproduced the corrected outputs byte-for-byte. The fee discrepancy is real but economically small relative to the strategy gap.

The scope explicitly says to move on rather than expand a fee grid when this correction is immaterial. Therefore this continuation does **not** rerun ChatGPT, Dumate or WorkBuddy merely to apply the same stamp patch. An attempted exact ChatGPT native replay was stopped after 300 seconds with no result and is excluded from all conclusions.

## Remaining correctness boundary

Fee semantics and qfq/corporate-action basis are now classified rather than silently mixed. They do not produce a new accepted reference or a quant production improvement. The useful remaining correctness work is narrower: inspect risk/protection state semantics against authenticated source + ledger evidence and run a fixed correction only if a concrete issue actually binds. Otherwise the remaining shortfall is an economic-design problem, and the exhausted strategy budgets cannot be reset by renaming correctness work.

Economic acceptance remains `NOT_MET`; production delivery remains `NOT_COMPLETE`; PR #1 remains ineligible for merge.
