# Independent A-share technology research contract

The implementation in this repository is newly written. The four projects in `ychenracing/trades` are read-only references and external comparators, not a source-code base. No reference strategy modules, copied functions, inherited engine classes, or stock-specific tuning tables may enter the production package.

## Scope

Cash-only, long-only A-share technology decision support. Signals use completed sessions and are actionable no earlier than the following session. No broker connection, leverage, shorting, automatic orders, or reliance on consumption/pharmaceutical themes. Requested measurement: 2023-01-03 through the latest verified completed session available on 2026-09-13. Earlier history must not be used for training or warm-up.

## Independent design

Separate causal trend selection, portfolio risk budgeting, realistic execution/accounting, and evidence production. Prefer one documented global parameter set and no per-symbol or calendar-date strategy branches. Trading frequency is constrained by holding hysteresis and order materiality, not by suppressing protective exits. Recovery must use fresh observable evidence, not knowledge of the end of a historical crash.

## Evidence

Freeze source, configuration, datasets, universe, initial capital, costs, metric definitions, random seed, and runtime before final measurement. Separate engineering correctness, data validity, native reference results, normalized comparisons, and economic acceptance. Retain failures. Never present a proxy comparator as an exact replay of an original implementation. Never turn incomplete date or universe coverage into a pass.

Measure full-period and requested bull/crash subperiods, buy-and-hold and equal-weight baselines, original pools, single-name removal, joint leader removal, sector removal, deterministic size-stratified subsets, cost/execution stress, prefix invariance, and stability around the selected parameter setting. Finite sampled tests are not a proof of all possible subsets or future optimality. Universal superiority remains unverified unless actually established.

## Delivery

Implement and validate without further approval requests within this scope. Publish independently written code, useful comments, operating documentation, reproducible research entry points, and an explicit verified/failed/blocked status to main. Do not label unproven economics accepted. Keep ordinary CI short, offline, and limited to critical correctness checks. Large historical research is not a required CI matrix.

## Reference identity

trades main read on 2026-09-13: `5575d1b1b79fab405b92cfb7d164c7054a8bd826`.
A downloaded source artifact from `c3d838ce6349f8bdf0ec2988836212563aaa10bc` contains the four reference projects. GitHub's complete comparison between these commits shows no changes under their four directories. Its additional removed projects are not implementation sources.
