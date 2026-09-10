"""
Assembles the daily feature matrix X_t (roadmap section 25, Figure 19:
"Raw APIs/Files -> SQL -> Release-Date Alignment -> Feature Engineering ->
Daily Feature Matrix X_t"). SPY trading days are the base calendar; every
other feature is as-of joined onto it so nothing uses information from
after that day.

Not one of the files named in the PDF's suggested layout (section 22) --
added as the natural place to wire macro/market/prediction-market features
together into one table, since the PDF splits them into 3 separate feature
modules but something has to combine them.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "daily_features.csv"

# src/database and src/features are plain directories (no package setup),
# so a normal `import database` needs their parents on sys.path first.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "database"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import database  # noqa: E402
import macro_features  # noqa: E402
import market_features  # noqa: E402
import prediction_market_features  # noqa: E402


def _asof_macro_feature(conn: sqlite3.Connection, trading_days: pd.DataFrame, series_id: str, out_name: str) -> pd.Series:
    series = macro_features.load_series(conn, series_id)
    return macro_features.asof_join(trading_days, series, out_name)


def build(conn: sqlite3.Connection) -> pd.DataFrame:
    trading_days = market_features.load_spy(conn)[["date"]]

    spy_feat = market_features.spy_features(market_features.load_spy(conn))
    vix_feat = market_features.load_vix(conn)

    cpi = macro_features.load_series(conn, "cpi")
    cpi_yoy = macro_features.asof_join(
        trading_days, macro_features.yoy_series(cpi).rename(columns={"yoy": "value"}), "cpi_yoy"
    )
    cpi_surprise = macro_features.asof_join(
        trading_days, macro_features.mom_change_series(cpi).rename(columns={"mom_change": "value"}), "cpi_surprise"
    )

    ppi = macro_features.load_series(conn, "ppi")
    ppi_yoy = macro_features.asof_join(
        trading_days, macro_features.yoy_series(ppi).rename(columns={"yoy": "value"}), "ppi_yoy"
    )

    unemployment_rate = _asof_macro_feature(conn, trading_days, "unemployment_rate", "unemployment_rate")
    yield_2y = _asof_macro_feature(conn, trading_days, "yield_2y", "yield_2y")
    yield_10y = _asof_macro_feature(conn, trading_days, "yield_10y", "yield_10y")

    fed_feat = prediction_market_features.fed_cut_probability_features(conn, trading_days)

    out = trading_days.copy()
    out = out.merge(spy_feat, on="date", how="left")
    out = out.merge(vix_feat, on="date", how="left")
    out["cpi_yoy"] = cpi_yoy.values
    out["cpi_surprise"] = cpi_surprise.values
    out["ppi_yoy"] = ppi_yoy.values
    out["unemployment_rate"] = unemployment_rate.values
    out["yield_2y"] = yield_2y.values
    out["yield_10y"] = yield_10y.values
    out["yield_spread_10y_2y"] = out["yield_10y"] - out["yield_2y"]
    out = out.merge(
        fed_feat[["date", "fed_cut_probability", "fed_cut_probability_delta"]],
        on="date", how="left",
    )

    schema_cols = [
        "date", "spy_return_1d", "spy_return_5d", "realized_vol",
        "fed_cut_probability", "fed_cut_probability_delta",
        "cpi_yoy", "cpi_surprise", "ppi_yoy", "unemployment_rate",
        "yield_2y", "yield_10y", "yield_spread_10y_2y", "vix",
    ]
    return out[schema_cols]


def save(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM daily_features")
    df.to_sql("daily_features", conn, if_exists="append", index=False)
    conn.commit()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    conn = database.get_connection()
    df = build(conn)
    save(df, conn)
    print(f"{len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"-> {database.DB_PATH} (table daily_features)")
    print(f"-> {OUT_CSV}")
    print("\nNaN counts per column:")
    print(df.isna().sum())
    print("\nLast 5 rows:")
    print(df.tail())
    conn.close()
