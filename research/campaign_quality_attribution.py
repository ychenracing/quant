"""Read-only cross-scope attribution for an owned campaign-quality state.

The diagnostic observes actual funded campaigns from the unchanged two-position
price book.  At the close of the fifth and tenth market sessions after the
first fill, it asks whether the owned security has positive excess log return
over the frozen CSI 300 observation at both horizons and a positive absolute
ten-session return.  Only information available by that fixed review close is
used to classify a campaign; later cash PnL is a retrospective label only.

Campaigns closed before the tenth-session review and campaigns still open at
the frozen cutoff are reported separately and cannot make the state pass.
This script does not alter targets, replay a candidate, or create an entry or
exit rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from techquant.config import Config
from techquant.data import file_hash, load_market
from techquant.engine import run
from techquant.evidence import metrics
from research.expectation_study import scopes
from research.ledger_attribution import attribute
from research.trend_book import Owner, Parameters


CUTOFF = "2025-12-31"
BENCHMARK = "sh000300"
FAST_SESSIONS = 5
FULL_SESSIONS = 10


def load_benchmark(root: Path, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row["symbol"] == BENCHMARK]
    if len(records) != 1 or records[0].get("role") != "observation":
        raise ValueError("frozen CSI 300 observation is missing or ambiguous")
    record = records[0]
    info = record["raw"]
    path = (root / info["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("benchmark path escapes frozen input root")
    if file_hash(path) != info["sha256"]:
        raise ValueError("benchmark SHA256 mismatch")
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if len(frame) != info["rows"]:
        raise ValueError("benchmark row count mismatch")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("benchmark sessions must be unique and increasing")
    selected = frame.reindex(calendar)
    if selected[["open", "close"]].isna().any().any():
        raise ValueError("benchmark does not cover the selection calendar")
    return selected, {
        "symbol": BENCHMARK,
        "name": record.get("quote_identity", {}).get(BENCHMARK, [None, None])[1],
        "csv_sha256": info["sha256"],
        "manifest_sha256": file_hash(manifest_path),
    }


def safe_share(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def group_summary(rows: list[dict], state: bool) -> dict:
    chosen = [row for row in rows if row["persistent_relative_quality"] is state]
    pnl = [row["pnl_cny"] for row in chosen]
    return {
        "campaigns": len(chosen),
        "wins": int(sum(value > 0 for value in pnl)),
        "win_rate": float(np.mean([value > 0 for value in pnl])) if pnl else None,
        "median_pnl_cny": float(np.median(pnl)) if pnl else None,
        "total_pnl_cny": float(sum(pnl)),
        "positive_pnl_cny": float(sum(max(0.0, value) for value in pnl)),
        "negative_pnl_cny": float(sum(min(0.0, value) for value in pnl)),
    }


def classify_campaign(market, benchmark: pd.DataFrame, row) -> tuple[str, dict]:
    entry = pd.Timestamp(row.entry)
    entry_i = int(market.calendar.get_loc(entry))
    fast_i = entry_i + FAST_SESSIONS - 1
    full_i = entry_i + FULL_SESSIONS - 1
    if full_i >= len(market.calendar):
        return "RIGHT_CENSORED_BEFORE_REVIEW", {}
    fast_date = market.calendar[fast_i]
    full_date = market.calendar[full_i]
    if row.exit != "OPEN" and pd.Timestamp(row.exit) <= full_date:
        return "CLOSED_BEFORE_REVIEW", {
            "review_date": str(full_date.date()),
        }
    if row.exit == "OPEN":
        outcome = "RIGHT_CENSORED_AFTER_REVIEW"
    else:
        outcome = "CLOSED_REVIEWED"

    frame = market.frames[row.symbol]
    prices = frame["close"].reindex(market.calendar).ffill()
    entry_open = float(frame.loc[entry, "open"])
    fast_close = float(prices.iloc[fast_i])
    full_close = float(prices.iloc[full_i])
    benchmark_open = float(benchmark.loc[entry, "open"])
    benchmark_fast = float(benchmark.loc[fast_date, "close"])
    benchmark_full = float(benchmark.loc[full_date, "close"])
    values = [entry_open, fast_close, full_close, benchmark_open,
              benchmark_fast, benchmark_full]
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"invalid review price for {row.symbol} entered {entry.date()}")

    stock_fast = fast_close / entry_open - 1.0
    stock_full = full_close / entry_open - 1.0
    benchmark_fast_return = benchmark_fast / benchmark_open - 1.0
    benchmark_full_return = benchmark_full / benchmark_open - 1.0
    excess_fast = float(np.log(fast_close / entry_open)
                        - np.log(benchmark_fast / benchmark_open))
    excess_full = float(np.log(full_close / entry_open)
                        - np.log(benchmark_full / benchmark_open))
    state = bool(excess_fast > 0 and excess_full > 0 and stock_full > 0)
    return outcome, {
        "fast_review_date": str(fast_date.date()),
        "review_date": str(full_date.date()),
        "stock_return_fast": float(stock_fast),
        "stock_return_full": float(stock_full),
        "benchmark_return_fast": float(benchmark_fast_return),
        "benchmark_return_full": float(benchmark_full_return),
        "excess_log_return_fast": excess_fast,
        "excess_log_return_full": excess_full,
        "persistent_relative_quality": state,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    catalog = json.loads((args.source / "research/catalog.json").read_text())
    full = load_market(args.data, supplement=args.supplement, sectors=catalog["sectors"])
    market = full.prefix(CUTOFF)
    benchmark, benchmark_identity = load_benchmark(args.data, market.calendar)
    output = {
        "status": "READ_ONLY_CAMPAIGN_QUALITY_ATTRIBUTION_NOT_CANDIDATE",
        "hypothesis": (
            "A funded campaign that beats the frozen CSI 300 over both the first "
            "five and ten sessions, while remaining positive in absolute terms at "
            "session ten, identifies compoundable ownership rather than a broad-pool "
            "distractor."
        ),
        "fixed_rule": {
            "origin": "first actual funded open",
            "fast_review_sessions": FAST_SESSIONS,
            "full_review_sessions": FULL_SESSIONS,
            "state": (
                "positive stock-minus-benchmark excess log return at both reviews "
                "and positive absolute stock return at the full review"
            ),
            "outcome_label": "actual closed-campaign cash PnL after the fixed review",
            "exclusions": [
                "campaigns closed before full review",
                "campaigns open at frozen cutoff",
            ],
        },
        "advance_rule": (
            "In every scope both states must have at least three closed reviewed "
            "campaigns; quality-state median PnL must be positive and its win rate "
            "must exceed the false state; it must retain at least 50% of reviewed "
            "positive PnL while the false state contains at least 50% of reviewed "
            "negative-PnL magnitude."
        ),
        "selection_end": CUTOFF,
        "benchmark": benchmark_identity,
        "full_data_sha256": full.fingerprint(),
        "selection_data_sha256": market.fingerprint(),
        "source_sha256": file_hash(Path(__file__)),
        "scopes": [],
    }

    for scope, symbols in scopes(market, catalog).items():
        scoped = market.subset(symbols)
        result = run(
            scoped,
            Config(),
            policy_factory=lambda selected, _config: Owner(selected, Parameters(2)),
        )
        _, episodes, ledger = attribute(scoped, result)
        reviewed, short, censored = [], [], []
        for episode in episodes.sort_values(["entry", "symbol"]).itertuples():
            outcome, review = classify_campaign(scoped, benchmark, episode)
            base = {
                "symbol": episode.symbol,
                "entry": episode.entry,
                "exit": episode.exit,
                "pnl_cny": float(episode.pnl),
                "return_on_buys": float(episode.return_on_buys),
                "buy_notional_cny": float(episode.buy_notional),
            }
            item = {**base, **review}
            if outcome == "CLOSED_REVIEWED":
                reviewed.append(item)
            elif outcome == "CLOSED_BEFORE_REVIEW":
                short.append(item)
            else:
                censored.append({**item, "censoring": outcome})

        quality = group_summary(reviewed, True)
        other = group_summary(reviewed, False)
        positive_total = sum(max(0.0, row["pnl_cny"]) for row in reviewed)
        negative_total = -sum(min(0.0, row["pnl_cny"]) for row in reviewed)
        retained_positive = safe_share(quality["positive_pnl_cny"], positive_total)
        rejected_negative = safe_share(-other["negative_pnl_cny"], negative_total)
        criteria = {
            "minimum_support": quality["campaigns"] >= 3 and other["campaigns"] >= 3,
            "positive_quality_median": (
                quality["median_pnl_cny"] is not None and quality["median_pnl_cny"] > 0
            ),
            "quality_win_rate_exceeds_other": (
                quality["win_rate"] is not None and other["win_rate"] is not None
                and quality["win_rate"] > other["win_rate"]
            ),
            "retains_half_positive_pnl": (
                retained_positive is not None and retained_positive >= 0.5
            ),
            "other_contains_half_negative_pnl": (
                rejected_negative is not None and rejected_negative >= 0.5
            ),
        }
        output["scopes"].append({
            "scope": scope,
            "control": {
                **metrics(result),
                "average_exposure": float(result.equity.exposure.mean()),
                "ledger": ledger,
            },
            "closed_reviewed_campaigns": len(reviewed),
            "closed_before_review_campaigns": len(short),
            "right_censored_campaigns": len(censored),
            "closed_before_review_pnl_cny": float(sum(row["pnl_cny"] for row in short)),
            "quality_state": quality,
            "other_state": other,
            "quality_positive_pnl_retained_share": retained_positive,
            "other_negative_pnl_contained_share": rejected_negative,
            "criteria": criteria,
            "scope_pass": all(criteria.values()),
            "reviewed_rows": reviewed,
            "closed_before_review_rows": short,
            "right_censored_rows": censored,
        })

    output["advance_to_preregistration"] = all(
        row["scope_pass"] for row in output["scopes"]
    )
    output["decision"] = (
        "ADVANCE_TO_PREREGISTRATION"
        if output["advance_to_preregistration"]
        else "REJECTED_BEFORE_IMPLEMENTATION"
    )
    rendered = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
