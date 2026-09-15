# Dumate common-five next-session-open execution diagnostic

This is a correctness-only comparator diagnostic under original quant PR #1. It consumes no economic strategy budget and does not modify quant production code, thresholds, universe, frozen data, or the private reference.

The authenticated Dumate native common-five replay was first reproduced exactly on economic outputs: wealth **1.465009570726x**, MDD **20.686106%**, 237 trade rows; archived equity/trades/settings hashes match. Static audit had already established that all 237 native fills use same-session close with configured slippage while the same session's close/risk/tradeability state can contribute to decisions.

The fixed diagnostic changes only that execution class: a completed-session decision on D executes no earlier than the immediate next global trading-session open. Native call ordering, selection/risk logic, thresholds, costs and slippage stay unchanged. Terminal inventory is marked to the final close rather than force-liquidated beyond the sample.

Measured once (and repeated byte-for-byte deterministically): corrected wealth **1.959828243839x**, MDD **19.151523%**, 243 trade rows. Relative terminal wealth changes by **+33.7758%**; MDD changes by **-1.5346 percentage points**. This direction is diagnostic, not a claim that timing corrections improve strategies generally.

Independent replay found zero T+1 opening-inventory violations, zero decision/fill ordering violations, exact next-open pricing to <=1.14e-13, cash-after error <=9.32e-10 and daily NAV error <=1.87e-09. Final positions remain 2,200 units of 300394 and 200 units of 603986 and are marked to 2026-09-11 close.

The required decline windows are not improved uniformly: 2026-06-22..08-31 changes from -2.2011%/3.0346% return/MDD to -2.9083%/4.2244%; 2026-07-01..08-31 changes from -2.1580%/2.3513% to -2.5153%/3.1507%. Full-period corrected MDD remains **19.1515%**, still above the original 18% ceiling.

Interpretation: Dumate's native same-close backtest is materially sensitive to execution timing; on this fixed normalization, terminal wealth rises but the two 2026 decline-window losses deepen. Preserve native and corrected results separately. This does not lower the original benchmark target and is not quant production acceptance.
