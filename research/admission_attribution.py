"""Attribute issued parent decisions using original filled inventory, never refits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.expectation_study import write_json


def admission_attribution(root: Path, market, out: Path):
    """Exact parent-fill replay of decision boundaries, without refitting models."""
    from techquant.evidence import load_result
    from techquant.policy import CloseObservation
    from research.issued_forecasts import IssuedForecasts
    from research.ledger_attribution import attribute
    from research.nonlinear import Owner as FundedOwner, Parameters as OwnershipParameters
    from research.observed_trend import Owner as TrendOwner, Parameters as TrendParameters
    issued = IssuedForecasts(root)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for scope in ('union', 'chatgpt_5', 'joint_optical_leader_removal'):
        path = root / 'evaluation/runs' / (scope + '_pathwise')
        identity = json.loads((path / 'identity.json').read_text())
        result = load_result(path, expected=identity)
        m = market.subset(identity['universe'])
        f, origin = issued.load(m)
        owner = TrendOwner(m, TrendParameters(60, True), learner=FundedOwner(
            m, OwnershipParameters(20, .5, 4), prediction=f))
        days, episodes, ledger = attribute(m, result)
        days.to_csv(out / (scope + '_symbol_days.csv'), index=False, float_format='%.17g')
        episodes.to_csv(out / (scope + '_episodes.csv'), index=False, float_format='%.17g')
        fills = {str(d.date()): [] for d in m.calendar}
        for order in result.orders:
            if order['status'] == 'FILLED':
                fills[order['date']].append(order)
        units = np.zeros(len(m.symbols)); rows = []; maximum = 0.
        for i, (date, equity) in enumerate(result.equity.iterrows()):
            for order in fills[str(date.date())]:
                units[m.symbols.index(order['symbol'])] += order['units'] * (1 if order['side'] == 'BUY' else -1)
            units[np.abs(units) < 1e-8] = 0.
            weights = np.nan_to_num(units * f.price[i], nan=0.) / equity.nav
            observation = CloseObservation.from_inventory(i, str(date.date()), float(equity.nav),
                                                           float(equity.cash), units, weights)
            request = owner.decide(observation)
            difference = float(np.max(np.abs(request.weights - result.targets.loc[date].to_numpy())))
            maximum = max(maximum, difference)
            if difference > 1e-12:
                raise ValueError('parent actual-fill decision cannot be exactly reproduced')
            held = units > 1e-10; score = f.expected[i]
            ranked = sorted(np.flatnonzero(f.ready[i] & (score > .005)),
                            key=lambda j: (-score[j], m.symbols[j]))
            entrants = [j for j in ranked if not held[j] and not owner.inner.readmit[j]
                        and f.momentum5[i,j] > 0 and f.price[i,j] > f.ema20[i,j] and f.tail[i,j] < .5]
            slots = max(0, min(4, len(held)) - int(held.sum()))
            allowed = owner.s.entry[i] & owner.s.market[i]
            denied = [m.symbols[j] for j in entrants[:slots] if not allowed[j]]
            alternatives = [m.symbols[j] for j in entrants[slots:] if allowed[j]]
            possible = (f.ready[i] & ~held & ~owner.inner.readmit & (f.momentum5[i] > 0)
                        & (f.price[i] > f.ema20[i]) & (f.tail[i] < .5) & allowed)
            no_score = [m.symbols[j] for j in np.flatnonzero(possible & (score <= .005))]
            funded = equity.cash / equity.nav >= .01 and not owner.inner.exit_pending.any() and slots > 0
            rows.append(dict(date=str(date.date()), cash_fraction=equity.cash/equity.nav,
                exposure=equity.exposure, slots=slots, pending_exit=bool(owner.inner.exit_pending.any()),
                market_admission=bool(owner.s.market[i]),
                shadowed_admissible=','.join(alternatives) if funded and denied else '',
                denied_preallocation=','.join(denied) if funded else '',
                price_ready_below_score=','.join(no_score) if funded else '', reason=request.reason))
        frame = pd.DataFrame(rows)
        frame.to_csv(out / (scope + '_decision_boundaries.csv'), index=False, float_format='%.17g')
        before = frame.date < '2026-01-01'
        summary.append(dict(scope=scope, origin=origin, ledger=ledger,
            all_decisions_match_original=True, maximum_weight_difference=maximum,
            shadowed_admissible_days_before_2026=int((before & frame.shadowed_admissible.ne('')).sum()),
            shadowed_admissible_days_full=int(frame.shadowed_admissible.ne('').sum()),
            price_ready_below_score_days_before_2026=int((before & frame.price_ready_below_score.ne('')).sum())))
    write_json(out / 'summary.json', {'status':'DIAGNOSTIC_NOT_ACCEPTANCE','scopes':summary})
