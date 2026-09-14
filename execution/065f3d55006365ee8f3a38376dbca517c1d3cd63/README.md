# Execution sizing correction and measured account audit

Source: `065f3d55006365ee8f3a38376dbca517c1d3cd63`; exact tree `af7dadedb5d8328f0c2fb3203721785bbc55ebf3`; parent `18901a431247a4f73aff08e73f2d0a7b3025de44`. This preserves the concurrent observed-admission implementation, all frozen families, and main.

The eight-line engine correction reapplies the existing ordinary 1% whole-account opening-NAV floor after capacity, lot rounding and affordability clipping. It does not lower thresholds or filter protective reductions. Five new regressions were observed red then green. The bundle retains the initial test-fixture error as well as the corrected failing regressions; no failed evidence was removed. Integrated verification: 82 offline tests and 142 research tests passed. Exact-source hosted correctness run 34869713149 also completed successfully.

Read-only audit: 72 actual saved accounts, zero recorded fills newly blocked by this guard; cash/NAV reconcile and maximum daily PnL residual is 1.4901161193847656e-08 CNY. The defect is real but does not explain the historical wealth shortfall. This is bounded path applicability, NOT an economic run at the corrected SHA. Original result identities, including their runtimes, remain unchanged.

Recovered studies: 34839270406/bf839c9, 34838336120/64fa068, 34866550377/a3217a9, and 34868483244/18901a4. All 555 top-manifest entries and exact source/dependency identities were checked. The first three remain DIAGNOSTIC_NOT_ACCEPTED. The last is CORE_NONREGRESSION_ONLY (zero pre-2026 deficit), not full economic acceptance. Its common-five full-period wealth is 10.598717 versus incumbent 10.824786. Retrospective filled-inventory attribution is included; it is not fresh out-of-sample or a counterfactual profit claim.

## Exact raw recovery

From this directory, concatenate `verification.json.xz.part000` through `part003` in manifest order. Check each part SHA256 and size, then the assembled SHA256 `97898d142eee0123304dc6683e93fbb9db8b41a772f42dd0d9a3fd9c0eacb14e`. Decompress using Python standard-library `lzma.decompress`, decode UTF-8 and parse JSON. Every `files[name].content` is the exact original UTF-8 raw text; verify its SHA256 against both the embedded value and manifest.json before writing it. Preserve all 16 files, including failed-test logs, audit scripts, full account-audit JSON, source patch, recovery verification and retrospective attribution. No market input or private trades source is republished in this bundle.

Engineering verification is not economic acceptance. Main/PR #1 must not be promoted on this record. Any continued economic candidate needs its own applicable identity-bound evidence and the original joint acceptance; historical results must never be relabelled as new-HEAD evidence.
