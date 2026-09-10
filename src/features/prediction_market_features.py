"""
Fed-cut probability feature: for each trading day, "the market" means
whichever FOMC meeting hasn't happened yet as of that day, and "probability
of a cut" means the combined chance of the cut_25 and cut_50 outcome
buckets for that specific meeting.

Two as-of decisions happen here, both backward/forward only relative to
the trading day -- never using information from after it:
1. Which meeting is "next" as of day t?  -> merge_asof(..., direction="forward")
   picks the smallest meeting_end_date >= t. This looks forward in *meeting
   dates*, not in *time* -- on any given day t we already know the full
   FOMC calendar (it's published a year ahead), so this isn't leakage.
2. What did the market think about that meeting, as of day t?
   -> merge_asof(..., direction="backward") on timestamp, same mechanism
   as macro_features.asof_join.
"""

import sqlite3

import pandas as pd


def load_fed_decision_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT meeting_end_date, outcome, timestamp, probability "
        "FROM polymarket_prices WHERE outcome IN ('cut_25', 'cut_50') "
        "ORDER BY meeting_end_date, timestamp",
        conn, parse_dates=["meeting_end_date", "timestamp"],
    )


def _cut_probability_by_meeting(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per (meeting_end_date, date): cut_25 + cut_50 probability,
    collapsing intraday snapshots to one value per calendar date first."""
    df = prices.copy()
    df["date"] = df["timestamp"].dt.normalize()
    daily = (
        df.sort_values("timestamp")
        .groupby(["meeting_end_date", "outcome", "date"], as_index=False)
        .last()[["meeting_end_date", "outcome", "date", "probability"]]
    )
    wide = daily.pivot_table(
        index=["meeting_end_date", "date"], columns="outcome", values="probability"
    ).reset_index()
    for col in ("cut_25", "cut_50"):
        if col not in wide.columns:
            wide[col] = 0.0
    wide["cut_probability"] = wide[["cut_25", "cut_50"]].fillna(0.0).sum(axis=1)
    return wide[["meeting_end_date", "date", "cut_probability"]]


def fed_cut_probability_features(conn: sqlite3.Connection, trading_days: pd.DataFrame) -> pd.DataFrame:
    prices = load_fed_decision_prices(conn)
    by_meeting = _cut_probability_by_meeting(prices)

    meetings = pd.DataFrame({
        "meeting_end_date": sorted(by_meeting["meeting_end_date"].unique())
    })
    active_meeting = pd.merge_asof(
        trading_days[["date"]].sort_values("date"), meetings,
        left_on="date", right_on="meeting_end_date", direction="forward",
    )

    out_rows = []
    for meeting_end_date, group in active_meeting.groupby("meeting_end_date"):
        meeting_prices = (
            by_meeting[by_meeting["meeting_end_date"] == meeting_end_date]
            .sort_values("date")
        )
        merged = pd.merge_asof(
            group[["date"]].sort_values("date"), meeting_prices[["date", "cut_probability"]],
            on="date", direction="backward",
        )
        merged["meeting_end_date"] = meeting_end_date
        out_rows.append(merged)

    result = pd.concat(out_rows, ignore_index=True).sort_values("date")
    result["fed_cut_probability_delta"] = result["cut_probability"].diff()
    return result.rename(columns={"cut_probability": "fed_cut_probability"})[
        ["date", "fed_cut_probability", "fed_cut_probability_delta", "meeting_end_date"]
    ]
