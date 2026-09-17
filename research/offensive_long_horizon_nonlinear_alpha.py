"""Causal nonlinear 60-session relative-return alpha for the offensive owner.

A single frozen HistGradientBoostingRegressor is refit every Config.rebalance
sessions from fully matured historical cohorts only.  The model predicts one
stable economic quantity: next-open-to-60-session-later-open stock log return
minus the same-cohort cross-sectional median.  It replaces only the alpha score
matrix inside the current campaign-peak champion; ownership and execution are
otherwise unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from techquant.config import Config
from techquant.data import Market, file_hash
from techquant.features import build_features
from research.offensive_campaign_peak_authority import (
    Owner as ChampionOwner,
    Parameters as ChampionParameters,
)
from research.quantity_obligation import preserve_trace, verify_trace


HORIZON = 60
RANDOM_STATE = 34


@dataclass(frozen=True)
class Parameters:
    enabled: bool

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")


def grid():
    return [Parameters(False), Parameters(True)]


def _log_return(close: pd.DataFrame, span: int) -> pd.DataFrame:
    return np.log(close / close.shift(span))


def _feature_tensor(market: Market) -> tuple[np.ndarray, object]:
    config = Config()
    base = build_features(market, config)
    close = market.panel("close").ffill()
    volume = market.panel("volume")
    log_close = np.log(close)
    one = log_close.diff()
    r5 = _log_return(close, 5)
    r20 = _log_return(close, 20)
    r60 = _log_return(close, 60)
    r120 = _log_return(close, 120)
    accel = r20 - r60 / 3.0
    high20 = np.log(close / close.rolling(20, min_periods=1).max())
    high60 = np.log(close / close.rolling(60, min_periods=1).max())
    vol20 = one.rolling(20, min_periods=5).std(ddof=0)
    vol60 = one.rolling(60, min_periods=10).std(ddof=0)
    volume_median = volume.rolling(20, min_periods=5).median()
    volume_ratio = np.log(volume / volume_median.replace(0, np.nan))

    rel20 = r20.sub(r20.median(axis=1), axis=0)
    rel60 = r60.sub(r60.median(axis=1), axis=0)
    rel120 = r120.sub(r120.median(axis=1), axis=0)

    sector_rel60 = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    sectors: dict[str, list[str]] = {}
    for symbol in market.symbols:
        sectors.setdefault(market.sectors.get(symbol, "unknown"), []).append(symbol)
    for symbols in sectors.values():
        median = r60[symbols].median(axis=1)
        sector_rel60.loc[:, symbols] = r60[symbols].sub(median, axis=0)

    score = np.asarray(base.score, dtype=float).copy()
    score[~np.isfinite(score)] = np.nan
    breadth = np.broadcast_to(np.asarray(base.breadth, dtype=float)[:, None], score.shape)
    market_dd = np.broadcast_to(np.asarray(base.market_dd, dtype=float)[:, None], score.shape)

    frames = [
        score,
        r5.to_numpy(dtype=float),
        r20.to_numpy(dtype=float),
        r60.to_numpy(dtype=float),
        r120.to_numpy(dtype=float),
        accel.to_numpy(dtype=float),
        high20.to_numpy(dtype=float),
        high60.to_numpy(dtype=float),
        vol20.to_numpy(dtype=float),
        vol60.to_numpy(dtype=float),
        volume_ratio.to_numpy(dtype=float),
        rel20.to_numpy(dtype=float),
        rel60.to_numpy(dtype=float),
        rel120.to_numpy(dtype=float),
        sector_rel60.to_numpy(dtype=float),
        breadth,
        market_dd,
    ]
    tensor = np.stack(frames, axis=2)
    tensor[~np.isfinite(tensor)] = np.nan
    return tensor, base


def _matured_training_rows(
    tensor: np.ndarray,
    ready: np.ndarray,
    execution_open: np.ndarray,
    refit_session: int,
    cadence: int,
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    origins: list[int] = []
    max_end = -1
    for origin in range(0, refit_session + 1, cadence):
        start = origin + 1
        end = start + HORIZON
        if end > refit_session or end >= len(execution_open):
            continue
        start_open = execution_open[start]
        end_open = execution_open[end]
        valid_return = (
            np.isfinite(start_open)
            & np.isfinite(end_open)
            & (start_open > 0)
            & (end_open > 0)
        )
        if not np.any(valid_return):
            continue
        raw = np.full(start_open.shape, np.nan, dtype=float)
        raw[valid_return] = np.log(end_open[valid_return] / start_open[valid_return])
        median = float(np.nanmedian(raw))
        target = raw - median
        usable = ready[origin] & np.isfinite(target)
        if not np.any(usable):
            continue
        rows.append(tensor[origin, usable])
        labels.append(target[usable])
        origins.append(origin)
        max_end = max(max_end, end)
    if not rows:
        return np.empty((0, tensor.shape[2])), np.empty(0), origins, max_end
    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0), origins, max_end


def walk_forward_scores(market: Market) -> tuple[np.ndarray, dict]:
    config = Config()
    tensor, base = _feature_tensor(market)
    execution_open = market.panel("open").to_numpy(dtype=float)
    learned = np.asarray(base.score, dtype=float).copy()
    model = None
    refits: list[dict] = []
    final_samples = 0

    for session in range(len(market.calendar)):
        if session % config.rebalance == 0:
            x, y, origins, max_end = _matured_training_rows(
                tensor,
                np.asarray(base.ready, dtype=bool),
                execution_open,
                session,
                config.rebalance,
            )
            if len(y):
                model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
                model.fit(x, y)
                final_samples = int(len(y))
                prediction = model.predict(x)
                if len(y) > 1 and np.std(y) > 0 and np.std(prediction) > 0:
                    target_rank = pd.Series(y).rank(method="average").to_numpy()
                    pred_rank = pd.Series(prediction).rank(method="average").to_numpy()
                    rank_corr = float(np.corrcoef(target_rank, pred_rank)[0, 1])
                else:
                    rank_corr = None
                refits.append({
                    "refit_session": int(session),
                    "max_origin": int(max(origins)),
                    "max_label_end": int(max_end),
                    "training_samples": int(len(y)),
                    "training_origins": int(len(origins)),
                    "horizon": HORIZON,
                    "training_rank_correlation": rank_corr,
                })
        if model is None:
            continue
        ready = np.asarray(base.ready[session], dtype=bool)
        row = np.full(len(market.symbols), -np.inf, dtype=float)
        if np.any(ready):
            row[ready] = model.predict(tensor[session, ready])
        learned[session] = row

    learned[~np.asarray(base.ready, dtype=bool)] = -np.inf
    audit = {
        "family": "offensive_long_horizon_nonlinear_alpha",
        "horizon": HORIZON,
        "cadence": int(config.rebalance),
        "random_state": RANDOM_STATE,
        "model": "HistGradientBoostingRegressor(defaults_except_random_state)",
        "fit_count": len(refits),
        "training_samples": final_samples,
        "refits": refits,
    }
    return learned, audit


class Owner:
    def __init__(self, market: Market, parameters: Parameters):
        if type(parameters) is not Parameters:
            raise ValueError("long horizon nonlinear alpha requires registered parameters")
        self.market = market
        self.parameters = parameters
        self.parent = ChampionOwner(market, ChampionParameters(True))
        self.learned_score = None
        self.audit = {
            "family": "offensive_long_horizon_nonlinear_alpha",
            "fit_count": 0,
            "training_samples": 0,
            "refits": [],
        }
        if parameters.enabled:
            learned, audit = walk_forward_scores(market)
            self.learned_score = learned
            self.audit = audit
            deep = self.parent.parent.base
            deep.features = replace(deep.features, score=learned)

    @property
    def trace(self):
        return self.parent.trace

    def decide(self, observation):
        return self.parent.decide(observation)

    def identity(self):
        root = Path(__file__).parent
        return {
            "name": "offensive_long_horizon_nonlinear_alpha",
            "parameters": asdict(self.parameters),
            "implementation_sha256": file_hash(Path(__file__)),
            "contract_sha256": file_hash(root / "offensive_long_horizon_nonlinear_alpha_contract.json"),
            "control_sha256": file_hash(root / "offensive_campaign_peak_authority.py"),
            "data_sha256": self.market.fingerprint(),
            "model": "sklearn.ensemble.HistGradientBoostingRegressor",
            "random_state": RANDOM_STATE,
            "horizon": HORIZON,
            "status": "RESEARCH_NOT_ACCEPTED",
        }


__all__ = ["Owner", "Parameters", "grid", "walk_forward_scores", "preserve_trace", "verify_trace"]
