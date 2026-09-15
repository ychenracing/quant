# Trae common-five historical stamp-duty correctness diagnostic

This fixed comparator-only check was preregistered before the corrected replay. It does not modify quant production code, strategy parameters, frozen data, universe, thresholds or either exhausted economic research budget.

The exact native Trae/common-five replay was first reproduced: wealth **10.674861915267x**, MDD **33.107753%**, 800 trade rows. Its trades and equity SHA256 values exactly match the authenticated archived stream.

The binding probe found **52 sells before 2023-08-28**, totaling CNY **32,863,762.70** notional. The native engine charges a constant 5bp sell stamp throughout the sample, while quant's frozen execution model records the historical 10bp rate before 2023-08-28 and 5bp thereafter. On the unchanged native trade path the missing 5bp would total CNY **16,431.88** before compounding/path effects. The first affected sale is 2023-02-20; its extra CNY **524.507615** stamp equals the first native-versus-corrected equity divergence.

Exactly one corrected replay changed only that historical sell-stamp schedule. All target generation, one-session shift, open fills, slippage, commission, transfer fee, minimum commission, lot rounding, price-limit rules, liquidity logic and risk state remain native. The corrected private source is not published. Corrected wealth is **10.638974540102x**, MDD **33.117142%**, 798 trade rows. Relative terminal wealth changes **-0.3362%**; MDD changes **0.00939 percentage points**.

A second corrected execution reproduced equity, trades, native frame and settings byte-for-byte. The fee correction therefore binds, but its measured magnitude is small: it does not explain the large quant-versus-strongest-reference gap, does not change the strongest reference, and does not justify a fee grid or lower target. Native and corrected results remain separate. Economic acceptance stays **NOT_MET** and PR #1 remains ineligible for merge.
