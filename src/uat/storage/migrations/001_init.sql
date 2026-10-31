-- Layer 1: raw. Append-only, never mutated.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS dim;
CREATE SCHEMA IF NOT EXISTS feat;
CREATE SCHEMA IF NOT EXISTS sig;
CREATE SCHEMA IF NOT EXISTS exec;
CREATE SCHEMA IF NOT EXISTS pos;
CREATE SCHEMA IF NOT EXISTS sys;

-- Point-in-time symbol directory snapshots (NASDAQ Trader files, one row per symbol per as_of).
CREATE TABLE IF NOT EXISTS raw.symbol_directory (
    as_of            DATE    NOT NULL,
    symbol           VARCHAR NOT NULL,
    exchange         VARCHAR,            -- NASDAQ | NYSE | AMEX | ARCA | BATS | IEX
    security_name    VARCHAR,
    is_etf           BOOLEAN,
    is_test          BOOLEAN,
    financial_status VARCHAR,            -- N normal, D deficient, E delinquent, Q bankrupt, H halted
    round_lot        INTEGER,
    source           VARCHAR,            -- nasdaqlisted | otherlisted | wayback:<ts>
    PRIMARY KEY (as_of, symbol)
);

-- RAW (unadjusted) OHLCV plus a cumulative split factor.
-- Price band / dollar-volume / fill simulation MUST use these raw columns.
-- Technical features multiply by adj_factor. Never store back-adjusted prices here.
CREATE TABLE IF NOT EXISTS raw.ohlcv_daily (
    symbol      VARCHAR NOT NULL,
    dt          DATE    NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      BIGINT,
    adj_factor  DOUBLE  DEFAULT 1.0,
    ingested_at TIMESTAMP,
    PRIMARY KEY (symbol, dt)
);

CREATE TABLE IF NOT EXISTS raw.corporate_actions (
    symbol VARCHAR NOT NULL,
    dt     DATE    NOT NULL,
    kind   VARCHAR NOT NULL,             -- split | dividend
    ratio  DOUBLE,
    PRIMARY KEY (symbol, dt, kind)
);

-- Layer 2: reference. Joined point-in-time via filed_date <= T, never period_end.
CREATE TABLE IF NOT EXISTS ref.sec_entity (
    cik                VARCHAR NOT NULL,
    ticker             VARCHAR NOT NULL,
    sic                VARCHAR,
    sic_desc           VARCHAR,
    latest_annual_form VARCHAR,
    is_foreign         BOOLEAN,
    as_of              DATE,
    PRIMARY KEY (cik, ticker)
);

CREATE TABLE IF NOT EXISTS ref.shares_outstanding (
    cik         VARCHAR NOT NULL,
    period_end  DATE,
    filed_date  DATE    NOT NULL,
    shares      BIGINT,
    PRIMARY KEY (cik, filed_date)
);

-- Dilution watch: a new shelf/ATM is the single most common way a penny runner dies.
CREATE TABLE IF NOT EXISTS ref.sec_filings (
    symbol      VARCHAR NOT NULL,
    cik         VARCHAR,
    form        VARCHAR NOT NULL,        -- S-1 | S-3 | 424B5 | 8-K | ...
    filed_date  DATE    NOT NULL,
    accession   VARCHAR NOT NULL,
    PRIMARY KEY (accession)
);

-- Layer 3: derived. Rejects are stored too, so universe churn is debuggable.
CREATE TABLE IF NOT EXISTS dim.universe_snapshot (
    as_of         DATE    NOT NULL,
    symbol        VARCHAR NOT NULL,
    tier          VARCHAR,               -- core | watch | rejected
    reject_reason VARCHAR,
    PRIMARY KEY (as_of, symbol)
);

CREATE TABLE IF NOT EXISTS sig.candidates (
    asof_dt       DATE    NOT NULL,
    symbol        VARCHAR NOT NULL,
    rank          INTEGER,
    score         DOUBLE,
    components    JSON,
    model_version VARCHAR,
    PRIMARY KEY (asof_dt, symbol)
);

CREATE TABLE IF NOT EXISTS exec.orders (
    order_id      VARCHAR NOT NULL,
    asof_dt       DATE,
    symbol        VARCHAR,
    side          VARCHAR,               -- BUY | SELL
    qty           INTEGER,
    limit_px      DOUBLE,
    status        VARCHAR,               -- NEW | PARTIAL | FILLED | CANCELLED | REJECTED
    kis_order_no  VARCHAR,
    reason        VARCHAR,               -- entry | tp1 | tp2 | trail | stop | time | thesis
    created_at    TIMESTAMP,
    updated_at    TIMESTAMP,
    PRIMARY KEY (order_id)
);

CREATE TABLE IF NOT EXISTS exec.fills (
    fill_id  VARCHAR NOT NULL,
    order_id VARCHAR NOT NULL,
    ts       TIMESTAMP,
    qty      INTEGER,
    px       DOUBLE,
    fee      DOUBLE,
    PRIMARY KEY (fill_id)
);

CREATE TABLE IF NOT EXISTS pos.positions (
    position_id    VARCHAR NOT NULL,
    symbol         VARCHAR NOT NULL,
    opened_at      TIMESTAMP,
    closed_at      TIMESTAMP,
    qty_open       INTEGER,
    qty_initial    INTEGER,
    avg_px         DOUBLE,
    stop_px        DOUBLE,
    tp_ladder      JSON,
    atr_at_entry   DOUBLE,
    atr_frozen     DOUBLE,                -- frozen at TP1; see exits.py
    hh_since_entry DOUBLE,
    hc_since_entry DOUBLE,
    state          VARCHAR,               -- OPEN | TP1 | TP2 | CLOSED
    PRIMARY KEY (position_id)
);

CREATE TABLE IF NOT EXISTS sys.equity_curve (
    dt         DATE NOT NULL,
    cash_usd   DOUBLE,
    mtm_usd    DOUBLE,
    equity_usd DOUBLE,
    hwm_usd    DOUBLE,
    PRIMARY KEY (dt)
);

CREATE TABLE IF NOT EXISTS sys.flags (
    key        VARCHAR NOT NULL,
    value      VARCHAR,
    updated_at TIMESTAMP,
    updated_by VARCHAR,
    PRIMARY KEY (key)
);

CREATE TABLE IF NOT EXISTS sys.job_runs (
    job          VARCHAR NOT NULL,
    session_date DATE    NOT NULL,
    started_at   TIMESTAMP,
    finished_at  TIMESTAMP,
    status       VARCHAR,
    detail       VARCHAR,
    PRIMARY KEY (job, session_date, started_at)
);

CREATE TABLE IF NOT EXISTS sys.schema_version (
    version    INTEGER NOT NULL,
    applied_at TIMESTAMP,
    PRIMARY KEY (version)
);
