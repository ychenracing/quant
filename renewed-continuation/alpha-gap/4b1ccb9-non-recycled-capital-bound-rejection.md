# Non-recycled campaign capital: rejected before implementation

Status: `REJECTED_BEFORE_IMPLEMENTATION`. This is a read-only local campaign
cash-flow bound, not a portfolio replay or an accepted strategy.

## Question and fixed counterfactual

The failed `basis_failure_exit` account released cash and then changed later
admissions. To isolate that mechanism, the unchanged two-position control was
observed without changing any decision. At the first two-close
acquisition-basis/EMA40 failure of each actually funded campaign, the diagnostic
assumed a full sale at the next session open with the frozen 10bp slippage and
commission/stamp schedule. The proceeds were then kept unavailable until that
same unchanged campaign's actual exit. No other security could use them.

Open campaigns at the 2025-12-31 selection cutoff would have been excluded;
none of the observed events was censored. Next-open limit and prior-only
liquidity capacity were recorded separately. This is still only a local bound:
it does not manufacture spendable portfolio cash or recompute NAV.

## Result

| scope | events | next-open fully executable | net cash preserved at original exit | positive preservation | sacrificed continuation profit | median quarantine |
|---|---:|---:|---:|---:|---:|---:|
| union | 15 | 13 | CNY -1,082,140 | CNY 351,744 | CNY 1,433,884 | 3 sessions |
| common-five | 8 | 7 | CNY -913,564 | CNY 200,328 | CNY 1,113,892 | 2.5 sessions |
| leader removal | 16 | 15 | CNY -632,765 | CNY 117,015 | CNY 749,780 | 5 sessions |

The executable-event subsets were also negative: CNY -619,146 / -905,280 /
-657,125. Therefore infeasible large exits are not what makes the aggregate
result fail.

The largest sacrificed recoveries were not small edge cases:

- union `sh688041`, signal 2024-11-26: CNY -529,467 by the unchanged
  2025-01-02 campaign exit;
- common-five `sh688008`, signal 2025-05-27: CNY -957,820 by the unchanged
  2025-08-26 campaign exit;
- removal `sh688072`, signal 2023-03-27: CNY -211,424 by the unchanged
  2023-04-21 campaign exit.

The exact JSON was produced twice byte-for-byte with SHA256
`d42525833392a05cc37d1e067487776de555e78bf691ba32484422f7bbcf7779`.
The diagnostic source SHA256 is
`f6dc3e35e5e5ab619a4fbdf34a235af499d909871e0f37a6cbedb0e92125e10f`.
Frozen full/pre-2026 data identities remain
`d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b`
and `894230b20361fb826d58b27987e87146a090b9e894e7abf091121486f5293157`.

Verification remained 87/87 ordinary tests and 414 research tests passed with
9 skips. The first research-test command omitted the existing `tests` import
path and failed during collection; rerunning with the repository's required
`PYTHONPATH=src:tests` completed successfully. No source behavior was changed.

## Decision

Do not implement, tune or rename a basis-failure cash quarantine, same-security
reserve, cooldown or released-cash sleeve. The diagnosis falsifies the claim
that cash recycling was the only cause of the prior candidate's failure:
the underlying event also cuts subsequent recoveries and compounders in all
three scopes.

No 2026 or final matrix was run. PR #1 remains unmerged and economic acceptance
remains `NOT_MET`.

