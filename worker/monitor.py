import os
import json
import time
from datetime import datetime, timedelta, date

import requests
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

AUTO_SIM_DAYS = int(os.environ.get("AUTO_SIM_DAYS", "5"))

def post_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {"content": message}
    requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=20)

def get_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def ensure_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            recorded_at TIMESTAMP NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            grade TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            entry DOUBLE PRECISION NOT NULL,
            stop DOUBLE PRECISION NOT NULL,
            target DOUBLE PRECISION NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            outcome TEXT DEFAULT '',
            outcome_r DOUBLE PRECISION,
            simulated_at TIMESTAMP,
            sim_note TEXT DEFAULT ''
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id BIGSERIAL PRIMARY KEY,
            logged_at TIMESTAMP NOT NULL,
            source TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            grade TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            entry DOUBLE PRECISION NOT NULL,
            stop DOUBLE PRECISION NOT NULL,
            target DOUBLE PRECISION NOT NULL,
            result TEXT NOT NULL,
            result_r DOUBLE PRECISION,
            notes TEXT DEFAULT ''
        );
        """))

def next_trading_days_prices(ticker: str, start_dt: date, days: int) -> pd.DataFrame:
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = (start_dt + timedelta(days=days + 10)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_str, end=end_str, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.head(days)

def simulate_signal_outcome(recorded_at: datetime, ticker: str, direction: str, entry: float, stop: float, target: float):
    start_day = (recorded_at.date() + timedelta(days=1))
    df = next_trading_days_prices(ticker, start_day, AUTO_SIM_DAYS)
    if df.empty:
        return ("NO_DATA", None, "No daily data returned for sim window.")

    R = abs(entry - stop)
    if R <= 0:
        return ("NO_DATA", None, "Invalid stop distance.")

    for _, day in df.iterrows():
        high = float(day.get("High"))
        low = float(day.get("Low"))

        if direction == "Bullish":
            hit_stop = (low <= stop)
            hit_target = (high >= target)
        else:
            hit_stop = (high >= stop)
            hit_target = (low <= target)

        if hit_stop and hit_target:
            return ("LOSS", -1.0, "Stop and target hit same day. Counted as loss (conservative).")
        if hit_stop:
            return ("LOSS", -1.0, "Stop hit first.")
        if hit_target:
            return ("WIN", abs(target - entry) / R, "Target hit first.")

    return ("NO_DATA", None, f"No stop/target hit in {AUTO_SIM_DAYS} trading days.")

def close_and_promote(engine):
    closed_count = 0
    promoted_count = 0

    with engine.begin() as conn:
        open_signals = conn.execute(text("""
            SELECT signal_id, recorded_at, ticker, direction, grade, timeframe, entry, stop, target
            FROM signals
            WHERE status = 'open'
            ORDER BY recorded_at ASC
        """)).fetchall()

    for s in open_signals:
        signal_id = s[0]
        recorded_at = s[1]
        ticker = s[2]
        direction = s[3]
        grade = s[4]
        timeframe = s[5]
        entry = float(s[6])
        stop = float(s[7])
        target = float(s[8])

        outcome, out_r, note = simulate_signal_outcome(recorded_at, ticker, direction, entry, stop, target)
        if outcome not in ("WIN", "LOSS"):
            continue

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE signals
                SET status='closed',
                    outcome=:outcome,
                    outcome_r=:out_r,
                    simulated_at=NOW(),
                    sim_note=:note
                WHERE signal_id=:signal_id
            """), {"outcome": outcome, "out_r": out_r, "note": note, "signal_id": signal_id})
            closed_count += 1

            conn.execute(text("""
                INSERT INTO trades (logged_at, source, ticker, direction, grade, timeframe, entry, stop, target, result, result_r, notes)
                VALUES (NOW(), 'auto-sim', :ticker, :direction, :grade, :timeframe, :entry, :stop, :target, :result, :result_r, :notes)
            """), {
                "ticker": ticker,
                "direction": direction,
                "grade": grade,
                "timeframe": timeframe,
                "entry": entry,
                "stop": stop,
                "target": target,
                "result": outcome,
                "result_r": out_r,
                "notes": f"Auto-sim: {note}"
            })
            promoted_count += 1

        post_discord(
            f"Greenlight update: **{ticker}** signal closed as **{outcome}**. "
            f"Entry {entry:.2f} | Stop {stop:.2f} | Target {target:.2f} | R {out_r if out_r is not None else 'n/a'}"
        )

    return closed_count, promoted_count

def main():
    engine = get_engine()
    ensure_tables(engine)
    closed, promoted = close_and_promote(engine)
    print(f"{datetime.now().isoformat(timespec='seconds')} closed={closed} promoted={promoted}")

if __name__ == "__main__":
    main()