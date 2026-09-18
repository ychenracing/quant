"""Native-reference input adapters and semantic no-op runtime optimizations."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from research.native import (
    install_chatgpt_semantic_noop_optimizations,
    native_input_frame,
)


class NativeInputAdapterTests(unittest.TestCase):
    def test_workbuddy_reindex_preserves_observations_and_marks_prelisting_missing(self):
        calendar = pd.date_range("2023-01-02", periods=5, freq="B")
        observed = calendar[2:]
        frame = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.2, 11.2, 12.2],
                "volume": [100.0, 200.0, 300.0],
                "raw_close": [20.0, 21.0, 22.0],
            },
            index=observed,
        )

        adapted = native_input_frame(frame, calendar, "workbuddy")
        self.assertTrue(adapted.index.equals(calendar))
        self.assertTrue(adapted.loc[calendar[:2]].isna().all().all())
        pd.testing.assert_frame_equal(
            adapted.loc[observed, ["open", "high", "low", "close", "volume"]],
            frame[["open", "high", "low", "close", "volume"]],
            check_freq=False,
        )
        np.testing.assert_array_equal(
            adapted.loc[observed, "amount"].to_numpy(),
            (frame.raw_close * frame.volume).to_numpy(),
        )

    def test_other_references_keep_the_observed_calendar(self):
        calendar = pd.date_range("2023-01-02", periods=5, freq="B")
        observed = calendar[2:]
        frame = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.2, 11.2, 12.2],
                "volume": [100.0, 200.0, 300.0],
                "raw_close": [20.0, 21.0, 22.0],
            },
            index=observed,
        )
        adapted = native_input_frame(frame, calendar, "chatgpt")
        self.assertTrue(adapted.index.equals(observed))


class ChatGptNoopOptimizationTests(unittest.TestCase):
    @staticmethod
    def native_classes():
        class Sleeve:
            score_calls = 0
            execute_calls = 0

            def _allocation_scores(self, data_map, date):
                type(self).score_calls += 1
                return {"new": float(pd.Timestamp(date).day)}

            def _execute_pending_signals(
                self, pending, data_map, date, date_to_pos, directions=None
            ):
                type(self).execute_calls += 1
                return list(pending)

        class Ensemble:
            authorize_calls = 0

            def __init__(self):
                self.cfg = {"max_positions": 2}

            @staticmethod
            def _held_portfolio_symbols(states):
                return {"held"}

            def _authorize_portfolio_buys(self, states, date):
                type(self).authorize_calls += 1
                return "original"

        return SimpleNamespace(
            SleeveBacktestEngine=Sleeve,
            BacktestEngine=Ensemble,
        )

    def test_same_sleeve_data_and_session_score_is_computed_once(self):
        native = self.native_classes()
        metadata = install_chatgpt_semantic_noop_optimizations(native)
        sleeve = native.SleeveBacktestEngine()
        data_map = {"new": pd.DataFrame(index=pd.date_range("2023-01-02", periods=2))}
        date = pd.Timestamp("2023-01-03")

        first = sleeve._allocation_scores(data_map, date)
        second = sleeve._allocation_scores(data_map, date)
        self.assertIs(first, second)
        self.assertEqual(native.SleeveBacktestEngine.score_calls, 1)
        sleeve._allocation_scores(data_map, pd.Timestamp("2023-01-04"))
        self.assertEqual(native.SleeveBacktestEngine.score_calls, 2)
        self.assertEqual(metadata["kind"], "semantic_noop_runtime_optimization")

    def test_empty_pending_queue_bypasses_unused_scores_and_execution(self):
        native = self.native_classes()
        install_chatgpt_semantic_noop_optimizations(native)
        sleeve = native.SleeveBacktestEngine()
        result = sleeve._execute_pending_signals(
            [], {}, pd.Timestamp("2023-01-03"), {}, frozenset({"buy"})
        )
        self.assertEqual(result, [])
        self.assertEqual(native.SleeveBacktestEngine.execute_calls, 0)
        self.assertEqual(native.SleeveBacktestEngine.score_calls, 0)

    def test_no_new_admission_candidate_reproduces_original_empty_rank_branch(self):
        native = self.native_classes()
        install_chatgpt_semantic_noop_optimizations(native)
        engine = native.BacktestEngine()
        events = []

        class Recorder:
            def _record_order_event(self, **event):
                events.append(event)

        held_buy = SimpleNamespace(direction="buy", symbol="held")
        unavailable_buy = SimpleNamespace(direction="buy", symbol="new")
        sell = SimpleNamespace(direction="sell", symbol="old")
        state = SimpleNamespace(
            pending=[(held_buy, "a"), (unavailable_buy, "b"), (sell, "c")],
            data_map={"new": pd.DataFrame(index=[pd.Timestamp("2023-01-04")])},
            sleeve=Recorder(),
        )
        engine._authorize_portfolio_buys([state], pd.Timestamp("2023-01-03"))

        self.assertEqual(native.BacktestEngine.authorize_calls, 0)
        self.assertEqual(state.pending, [(held_buy, "a"), (sell, "c")])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "rejected_portfolio_symbol_limit")
        self.assertEqual(events[0]["portfolio_max_positions"], 2)

    def test_real_new_candidate_delegates_to_frozen_authorizer(self):
        native = self.native_classes()
        install_chatgpt_semantic_noop_optimizations(native)
        engine = native.BacktestEngine()
        date = pd.Timestamp("2023-01-03")
        signal = SimpleNamespace(direction="buy", symbol="new")
        state = SimpleNamespace(
            pending=[(signal, "strategy")],
            data_map={"new": pd.DataFrame(index=[date])},
            sleeve=SimpleNamespace(),
        )
        result = engine._authorize_portfolio_buys([state], date)
        self.assertEqual(result, "original")
        self.assertEqual(native.BacktestEngine.authorize_calls, 1)


if __name__ == "__main__":
    unittest.main()
