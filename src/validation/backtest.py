"""
Turns out-of-sample direction predictions into a simple long/flat
backtest: long SPY when P(up) > 0.5, flat otherwise, vs. always-long
buy-and-hold as the benchmark.

Deliberately simple, and deliberately non-overlapping: each prediction is
for the SPY return over the next `horizon` trading days, but predictions
are made daily, so naively compounding every day's predicted return would
double-count overlapping holding periods. This module only ever backtests
a *subsampled*, non-overlapping sequence of (date, predicted probability,
realized forward return) rows -- callers are responsible for that
subsampling (see run_polymarket_comparison.py, which does it with
stride=horizon). No transaction costs are modeled -- see README Known
Limitations.
"""

import numpy as np
import pandas as pd


def run_backtest(dates: pd.Series, y_prob: np.ndarray, forward_return: np.ndarray) -> dict:
    """dates, y_prob, forward_return must already be non-overlapping and
    chronologically ordered. Returns strategy + buy-and-hold equity curves
    and summary stats."""
    position = (y_prob > 0.5).astype(float)
    strategy_return = position * forward_return

    equity = pd.DataFrame({
        "date": dates.reset_index(drop=True),
        "position": position,
        "strategy_return": strategy_return,
        "buy_hold_return": forward_return,
    })
    equity["strategy_equity"] = (1 + equity["strategy_return"]).cumprod()
    equity["buy_hold_equity"] = (1 + equity["buy_hold_return"]).cumprod()

    summary = {
        "n_trades": len(equity),
        "strategy_total_return": equity["strategy_equity"].iloc[-1] - 1,
        "buy_hold_total_return": equity["buy_hold_equity"].iloc[-1] - 1,
        "strategy_sharpe": _sharpe(equity["strategy_return"]),
        "buy_hold_sharpe": _sharpe(equity["buy_hold_return"]),
        "strategy_max_drawdown": _max_drawdown(equity["strategy_equity"]),
        "buy_hold_max_drawdown": _max_drawdown(equity["buy_hold_equity"]),
        "pct_time_long": position.mean(),
    }
    return {"equity": equity, "summary": summary}


def _sharpe(returns: pd.Series, periods_per_year: float = 252 / 5) -> float:
    """periods_per_year defaults to ~50.4, assuming a 5-trading-day
    holding period per trade (this backtest's default horizon)."""
    if returns.std() == 0 or returns.empty:
        return float("nan")
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()
