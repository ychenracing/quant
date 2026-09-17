# Quant PR #1 final acceptance closure — 2026-09-17

## Decision

Current final status is **NOT_MET / NOT_MERGE_ELIGIBLE**. PR #1 must remain Draft and unmerged under the current effective acceptance contract.

This closure does not lower the economic target, waive active-risk/recovery requirements, reopen the exhausted strategy-research budget, or reinterpret Native/Corrected reference outputs as equivalent execution.

## Bound identities

- production main at audit start/end: `9f74d4fbf2207063e2bdafc4216a7bc92a1c7255`
- research source: `7551af8af7506857c684b4cda79c885971ac8dd3`
- prior evidence head: `dd7d692d5783459b5fae677b65a2da83b207c3fd`
- PR #1 head: `efffb9a428258e921feaff125011cba46223bb18`
- frozen market SHA256: `d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b`
- frozen reference repository: `ychenracing/trades@5575d1b1b79fab405b92cfb7d164c7054a8bd826`

Frozen input transport was independently restored during this closure. Outer hosted artifact SHA256 was `a87e191252da3906550c8db6c74993124666de431a9583721717b2c5249bbb14`; concatenated frozen evidence archive SHA256 was `6f7864e8f60001790806dcedb2fb7abc5960fb992c936ee9ea2be02069f3f353`. The loaded market reproduced 34 symbols, 896 sessions, 2023-01-03 through 2026-09-11 and the frozen market fingerprint above.

## What is complete

1. Passive production-path source binding and 199-case historical comparator reuse proof are complete in `passive-final-audit/LATEST.json`. No new economic replay was needed for the 199 historical buy-hold accounts.
2. The passive adapter shares the existing engine buy-hold path; its core paired accounts are economically identical where proven. Window slices inherit the full account state; they are not fresh-cash window restarts.
3. Four frozen references have authenticated Native evidence and fixed Corrected diagnostics where an actual correctness defect was established. Native and Corrected results remain separate evidence layers.
4. Known active candidates and closed research families were rechecked rather than silently discarded. No already-measured candidate was found that establishes current final acceptance.
5. The previously easy-to-overlook `offensive_portfolio_leader_peak_authority` already has a hosted canonical result at source `9cf393b2708212543f1be81a5888a48efc0e6734`, run `35170077244`: `REJECTED_VS_CURRENT_ALPHA_CHAMPION`, `advance=false`, no full-history evaluation. A local diagnostic reproduction on current research source reached the same rejection and was not promoted to canonical evidence.

## Candidate eligibility

### Passive ownership

Full frozen-period accounts:

| scope | wealth | MDD | disposition |
|---|---:|---:|---|
| union | 9.102309x | 39.8791% | return screen can advance; final active capability not established |
| common-five | 29.062957x | 52.3895% | same |
| joint optical-leader removal | 5.716080x | 45.6913% | same |

Passive ownership does not actively sell, rotate the main line, reduce risk, or re-enter after a risk cut. Price recovery while continuously holding cannot be used as proof of active avoidance/recovery.

Its historical buy-hold cases also do not implement a strict per-security fee-inclusive budget quarantine: 148 cases have at least one security whose total spend exceeds its nominal initial allocation; maximum observed single-security overspend is about CNY 1,226.82. Account cash remains safe. This is a benchmark-semantics divergence, not a reason to silently rewrite a future active production candidate.

### Active candidates

The strongest already-measured active families do not close the risk/return contract simultaneously:

- `offensive_campaign_peak_authority`: full wealth about 10.7684x / 27.4832x / 2.45848x for union/common-five/leader-removal; full MDD about 44.04% / 41.20% / 65.10%. In 2026-06-22..08-31 it returns about -22.80% / +2.08% / -29.74% respectively.
- `offensive_alpha_decay_displacement`: full wealth about 7.38110x / 19.44495x / 2.96688x; full MDD about 42.57% / 43.45% / 63.46%. In 2026-06-22..08-31 it returns about -7.56% / -5.03% / -21.66%, with local MDD about 17.40% / 25.86% / 31.43%.
- `offensive_dominant_peak_authority` advanced an alpha-stage objective but did not establish final acceptance; full leader-removal wealth was about 2.45408x and retained the poor stress behavior of this family.
- `offensive_portfolio_leader_peak_authority` canonical run `35170077244` was rejected before full-history evaluation.
- positive-alpha, profitable-campaign, unified-compounder, funded-leader, portfolio-hurdle, shadow-opportunity, score-proportional, long-horizon nonlinear, matured-displacement and other recorded families are preserved as rejected or not-final-acceptance evidence; no neighboring-threshold rescue is authorized.

For context, authenticated reference stress evidence is materially better in the leader-removal risk window for at least Trae and Dumate: approximately +3.30% / 13.28% local MDD for Trae and +0.73% / 10.88% for Dumate during 2026-06-22..08-31. Therefore the current active candidates do not establish the requested practical risk-avoidance capability merely by emitting active events.

A single diagnostic attempt to layer the existing production `RiskState` onto the alpha-decay owner reduced some drawdowns but failed the fixed three-scope selection screen, including worse leader-removal drawdown. It was not promoted, no threshold variants were searched, and it is not canonical evidence.

## Four-reference comparison status

Authenticated common-five headline wealth (Native): ChatGPT 1.778233x, Trae 10.674862x, Dumate 1.465010x, WorkBuddy 20.221073x.

Fixed correctness diagnostics retained separately include Trae historical stamp-duty correction (10.638975x), Dumate next-open correction (1.959828x), and WorkBuddy inventory/timing correction (17.827946x). These are not a single normalized ranking.

The remaining `Normalized` / `Equivalent-execution` label cannot be honestly completed by replaying final target weights through the Quant engine: some reference decisions are stateful and depend on their own fill/inventory/risk state. Fees, liquidity, actual-share/corporate-action accounting and risk semantics are not all reducible to harmless post-processing. Corporate-action metadata needed for an actual-share normalization is not present in the frozen authenticated inputs. Therefore:

- Native: verified where the original reference supports the scope.
- Corrected: verified only for identified, fixed correctness defects.
- Normalized full-scope: not established.
- Equivalent-execution full-scope: not established and must not be fabricated.

Because no Quant candidate is currently final-eligible, rerunning a large comparator matrix would not cure the decisive acceptance failure and is not justified by the current closure contract.

## Engineering / production integration

Engineering correctness for the passive adapter, package build and installed CLI was already verified in the source-bound passive audit. Current production `src/` was not changed in this closure.

No research candidate was copied mechanically into production because none is final-eligible. The research owner chain remains research evidence rather than a second production engine. There is therefore no honest source-binding basis for marking PR #1 ready or merging it into main.

## Exact blocker and minimum next authorization

All work that does not require changing the current research contract is closed. The remaining blocker is not CI, Git transport, or an unknown historical result. It is a contract/product choice:

1. **Keep the active-risk / main-line-switch / recovery requirement.** Then a new, explicitly bounded strategy-research budget is required. It must be structurally new, preregistered, use only prefix-observable inputs, avoid per-pool splicing/neighbor tuning, and be rejected after its fixed budget if it cannot jointly satisfy return and risk evidence.
2. **Revise final product acceptance to passive ownership.** Then the user must explicitly accept passive high-drawdown behavior, account-level rather than strict per-security budget semantics, and the Native/Corrected reference evidence boundary instead of claiming full Equivalent-execution.

Absent one of those explicit contract changes, merging PR #1 would mislabel a known `NOT_MET` result as accepted, which is prohibited by the current task contract.
