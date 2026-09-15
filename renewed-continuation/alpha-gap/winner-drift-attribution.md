# Cross-sectional winner-drift attribution

This read-only audit joins the previously authenticated same-engine buy-hold and
quant accounts through 2025-12-31. For each symbol it reconstructs actual filled
cash flows, terminal adjusted-unit value, ownership sessions, mean account
weight and first-buy lag. It uses no reference-strategy source, produces no
targets and is not a candidate replay.

## Result

| Scope | Passive symbol PnL | Quant symbol PnL | Quant negative books | Passive-positive names negative in quant |
|---|---:|---:|---:|---:|
| union | CNY8.719m | CNY6.043m | CNY-1.929m | 11/29 |
| common-five | CNY36.055m | CNY9.006m | CNY-0.177m | 2/5 |
| leader removal | CNY4.095m | CNY2.117m | CNY-1.458m | 12/26 |

The five largest passive contributors explain 77.24% of positive passive PnL in
union. They produced CNY6.763m in passive and CNY7.791m in quant, so union's
largest winners were found and monetized. In common-five all five names were
passive winners, but passive PnL was CNY36.055m versus only CNY9.006m in quant;
median first-buy lag was 20 sessions and ownership/mean weight were materially
lower. In removal, the top five were closer (CNY2.618m passive versus CNY2.250m
quant), while negative quant books removed CNY1.458m.

The shared deficit is therefore not generic opportunity discovery. The small
pool loses winner duration/exposure; broad pools already monetize leading
winners but give back wealth through adverse campaign lifecycles. This is
consistent with prior fixed screens: generic leader retention helps
common-five but damages union/removal.

## Identity and limitation

- frozen data:
  `894230b20361fb826d58b27987e87146a090b9e894e7abf091121486f5293157`;
- audit source SHA256:
  `5a8c0e2a65d5c729d2c26095717dae89f2f42a196783edd56ed2fc0cdcab8774`;
- result SHA256:
  `146413a203629e1e28ff9d973406cf5a2fb251b8c47e471f108e5be14bd2c2cc`.

The result is descriptive and retrospective. PnL uses adjusted economic units,
not verified actual-share/corporate-action/tax accounting. Eventual passive PnL
must not enter any runtime rule.

## Continuation

Before implementation, compare acquisition-time prefix-observable state for the
11/12 broad-scope passive winners that became negative quant books with sustained
common-five winners. Require one fixed causal campaign-quality state to explain
the distinction in union, common-five and removal. Do not create named-security
rules, eventual-return labels, pool branches, another leader wrapper, extra gate
or neighboring search.
