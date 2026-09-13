# Execution state

Active implementation; economic superiority NOT accepted; main is still the initial LICENSE-only revision.

## Scope

Independently written A-share technology, cash-long-only daily research and human decision support. No reference strategy is the implementation base. Reference projects are read-only external comparators. Period starts 2023-01-03; signals after the close, execution no earlier than the next session; no broker integration, leverage or shorting.

## Preserved progress

The earlier independent-engine checkpoint is `07dd462690c2729984e46b26e3ceac27d0339a4e`; rejected bootstrap evidence remains under evidence/. This update preserves explicit execution intent, buy-and-hold semantics, protective small-order handling, bounded budget drift, progressive causal warm-up and risk-episode re-arming. Reported NAV and drawdown never reset. Nineteen compact offline tests passed locally in 0.601 seconds before this update. This is engineering evidence, not an economic pass.

The original 2023-2025 34-stock bootstrap returned wealth 2.6031143 with 19.6938% drawdown and 285 fills. It was rejected. Intermediate diagnostics after correctness fixes remained below buy-and-hold; they are not canonical acceptance evidence. No claim of improvement over all four projects is justified yet.

## Reference work

The four current strategy directories are chatgpt/turtle_dual, trae/glmcsm, dumate/momentum_rotation, workbuddy/track_trend. Their configured union has 34 names. The original trae pool identifier 24 actually contains 26 names. A read-only native harness is being prepared locally; it preserves original alpha, execution and fees, changing only data input, scope and capital. Native semantics differ and cannot be called equal-execution acceptance. Initial common-five native replays measured wealth 10.6748619 (trae), 20.2210730 (workbuddy) and 1.4650096 (dumate), through 2026-09-11. Chatgpt replay has not yet completed; interrupted attempts are retained. Full manifests and curves will be published with the finished harness.

## Frozen next research step

`research/protocol.json` fixes 12 candidates and the 2023-2025 numerical selection rule before executing the grid. Scope includes the union, common five and joint removal of the three optical leaders. 2026 is a retrospective stress evaluation, not a prospective untouched sample: its crash was already identified by the user. Do not expand the grid after observing 2026 results or silently relabel failures.

Remaining: finish strict data/hash audit, run and preserve the fixed grid, full pool/removal/subset/cost/delay/regime evaluation, native baselines, CLI/architecture/parameter/operations/results documentation, short CI, exact-source verification, then publish truthful working delivery to main. No economic conclusion is implied by publication.
