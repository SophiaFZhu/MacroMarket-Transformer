"""
The one property this whole project depends on: a trading day may never
see a macro value before its real release_date. Tests this directly on a
small synthetic series rather than the live pipeline, so it runs offline
and fast.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))
from macro_features import asof_join  # noqa: E402


def test_asof_join_never_uses_a_future_release():
    # A CPI-like series: two releases, real gap between observation and release.
    series = pd.DataFrame({
        "observation_date": pd.to_datetime(["2026-06-01", "2026-07-01"]),
        "release_date": pd.to_datetime(["2026-07-14", "2026-08-12"]),
        "value": [332.568, 332.813],
    })
    trading_days = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-11", "2026-08-12", "2026-08-13"])
    })

    result = asof_join(trading_days, series, "cpi")

    # Day before the July release: must still show the June value.
    assert result.iloc[0] == 332.568
    # Release day and after: July value becomes visible.
    assert result.iloc[1] == 332.813
    assert result.iloc[2] == 332.813


def test_asof_join_before_any_release_is_nan():
    series = pd.DataFrame({
        "observation_date": pd.to_datetime(["2026-07-01"]),
        "release_date": pd.to_datetime(["2026-08-12"]),
        "value": [332.813],
    })
    trading_days = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])})

    result = asof_join(trading_days, series, "cpi")

    assert pd.isna(result.iloc[0])
