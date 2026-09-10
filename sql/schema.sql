-- Raw macro observations, kept separate from when they became public.
-- observation_date = the period the number describes (e.g. "March 2026 CPI")
-- release_date      = the day the market could actually see this value
-- A feature table for date t must only join rows where release_date <= t.
CREATE TABLE IF NOT EXISTS macro_observations (
    series_id TEXT NOT NULL,
    observation_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (series_id, observation_date, release_date)
);

-- Polymarket expectation probabilities over time (continuous/intraday).
-- meeting_end_date + outcome are kept alongside market_id (rather than
-- normalized into a separate markets table) because feature engineering
-- needs to pick "whichever meeting is next as of date t" and sum the two
-- cut-bucket outcomes -- both need to be queryable without a join back to
-- an external file.
CREATE TABLE IF NOT EXISTS polymarket_prices (
    market_id TEXT NOT NULL,        -- full market question, e.g. "Fed decreases interest rates by 25 bps after January 2025 meeting?"
    meeting_end_date DATE NOT NULL, -- which FOMC meeting this market resolves on
    outcome TEXT NOT NULL,          -- one of: no_change, cut_25, cut_50, hike_25, hike_50
    timestamp DATETIME NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY (market_id, timestamp)
);

-- Daily SPY OHLCV.
CREATE TABLE IF NOT EXISTS spy_prices (
    date DATE PRIMARY KEY,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL
);

-- Daily VIX close (only the close is used as a feature; keep it minimal).
CREATE TABLE IF NOT EXISTS vix_prices (
    date DATE PRIMARY KEY,
    close REAL NOT NULL
);

-- Leakage-safe daily feature matrix X_t, built only from rows known by t.
CREATE TABLE IF NOT EXISTS daily_features (
    date DATE PRIMARY KEY,
    spy_return_1d REAL,
    spy_return_5d REAL,
    realized_vol REAL,
    fed_cut_probability REAL,
    fed_cut_probability_delta REAL,
    cpi_yoy REAL,
    cpi_surprise REAL,
    ppi_yoy REAL,
    unemployment_rate REAL,
    yield_2y REAL,
    yield_10y REAL,
    yield_spread_10y_2y REAL,
    vix REAL
);
