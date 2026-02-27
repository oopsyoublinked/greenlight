import os
import math
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests
import pytz
import yfinance as yf
from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

AUTO_SIM_DAYS = int(os.getenv("AUTO_SIM_DAYS", "5"))

# Universe: liquid ETFs + mega caps
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
    "JPM", "BAC", "NFLX", "COIN"
]

# Grading thresholds
A_CONF_MIN, A_RR_MIN, A_VOL_MAX = 7.5, 2.0, 6.5
B_CONF_MIN, B_RR_MIN, B_VOL_MAX = 6.5, 1.6, 7.5

# Chase thresholds (distance from EMA20)
CHASE_GREEN_MAX = 0.5
CHASE_YELLOW_MAX = 1.5

HTTP_TIMEOUT = 20

# =========================
# DISCORD
# =========================

def send_discord(msg: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=HTTP_TIMEOUT)
    except Exception as e:
        print("Discord send failed:", str(e))

def now_ct_str() -> str:
    try:
        central = pytz.timezone("US/Central")
        return datetime.now(central).strftime("%Y-%m-%d %I:%M %p CT")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# =========================
# INDICATORS
# =========================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def grade_from_metrics(conf: float, rr_val: float, vol: float) -> tuple[str, str]:
    if conf >= A_CONF_MIN and rr_val >= A_RR_MIN and vol <= A_VOL_MAX:
        return "A", "🟢"
    if conf >= B_CONF_MIN and rr_val >= B_RR_MIN and vol <= B_VOL_MAX:
        return "B", "🟡"
    return "C", "🔴"

def blended_score(conf: float, rr_val: float, vol: float) -> float:
    rr_component = min(rr_val, 3.0) / 3.0 * 10.0
    score = (conf * 0.55) + (rr_component * 0.30) + ((10.0 - vol) * 0.15)
    return float(np.clip(score, 0.0, 10.0))

def chase_meter(direction: str, last: float, ema20: float) -> tuple[str, str, float]:
    if ema20 <= 0:
        return "🟡", "Caution", 0.0

    if direction == "Bullish":
        dist_pct = (last - ema20) / ema20 * 100.0
    else:
        dist_pct = (ema20 - last) / ema20 * 100.0

    dist_pct = float(max(dist_pct, 0.0))
    if dist_pct < CHASE_GREEN_MAX:
        return "🟢", "Ideal", dist_pct
    if dist_pct < CHASE_YELLOW_MAX:
        return "🟡", "Slightly extended", dist_pct
    return "🔴", "Overextended", dist_pct

# =========================
# DATA FETCH
# =========================

def fetch_intraday_5m(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

# =========================
# PLAN MODEL
# =========================

@dataclass
class Plan:
    ticker: str
    timeframe: str
    direction: str
    grade: str
    dot: str
    score: float
    entry_now: float
    ema20: float
    stop: float
    target: float
    rr: float
    confidence: float
    vol_risk: float
    chase_dot: str
    chase_label: str
    chase_dist_pct: float

def build_plan_v1(ticker: str) -> Plan | None:
    df = fetch_intraday_5m(ticker)
    if df.empty or len(df) < 90:
        return None

    close = df["Close"]
    last = float(close.iloc[-1])

    ema20_s = close.ewm(span=20).mean()
    ema50_s = close.ewm(span=50).mean()
    ema20 = float(ema20_s.iloc[-1])
    ema50 = float(ema50_s.iloc[-1])

    r14_series = rsi(close, 14)
    r14 = float(r14_series.iloc[-1]) if not np.isnan(r14_series.iloc[-1]) else 50.0

    a_series = atr(df, 14)
    a = float(a_series.iloc[-1]) if not np.isnan(a_series.iloc[-1]) else last * 0.02

    atr_pct = a / last if last > 0 else np.nan
    vol_risk = float(np.clip((atr_pct * 100) / 2, 0, 10)) if not np.isnan(atr_pct) else 5.0

    bullish = (last > ema20) and (ema20 > ema50) and (r14 < 75)
    bearish = (last < ema20) and (ema20 < ema50) and (r14 > 25)
    if not bullish and not bearish:
        return None

    direction = "Bullish" if bullish else "Bearish"

    # Stop and target using ATR (targets about 2R)
    stop_dist = a * 1.2
    target_dist = stop_dist * 2.0
    stop = last - stop_dist if direction == "Bullish" else last + stop_dist
    target = last + target_dist if direction == "Bullish" else last - target_dist
    rr_val = abs(target - last) / max(abs(last - stop), 1e-9)

    # Momentum proxy: EMA20 slope over ~20 bars
    if len(ema20_s) >= 21:
        ema20_prev = float(ema20_s.iloc[-20])
    else:
        ema20_prev = ema20

    ema_slope = (ema20 - ema20_prev) / max(abs(ema20_prev), 1e-9)
    momentum_score = float(np.clip((ema_slope * 100) * 2 + 5, 0, 10))

    # Confidence: baseline + momentum + lower volatility preference
    confidence = float(np.clip((10.0 * 0.45 + momentum_score * 0.35 + (10 - vol_risk) * 0.20), 0, 10))

    grade, dot = grade_from_metrics(confidence, rr_val, vol_risk)
    if grade == "C":
        return None

    score = blended_score(confidence, rr_val, vol_risk)
    c_dot, c_label, c_dist = chase_meter(direction, last, ema20)

    return Plan(
        ticker=ticker,
        timeframe="2 to 5 days",
        direction=direction,
        grade=grade,
        dot=dot,
        score=score,
        entry_now=last,
        ema20=ema20,
        stop=float(stop),
        target=float(target),
        rr=float(rr_val),
        confidence=float(confidence),
        vol_risk=float(vol_risk),
        chase_dot=c_dot,
        chase_label=c_label,
        chase_dist_pct=float(c_dist),
    )

# =========================
# DB
# =========================

def get_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def ensure_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS plays (
                id BIGSERIAL PRIMARY KEY,
                computed_at TIMESTAMP NOT NULL,
                rank INT NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                grade TEXT NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                entry_now DOUBLE PRECISION NOT NULL,
                ema20 DOUBLE PRECISION NOT NULL,
                stop DOUBLE PRECISION NOT NULL,
                target DOUBLE PRECISION NOT NULL,
                rr DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                vol_risk DOUBLE PRECISION NOT NULL,
                chase_label TEXT NOT NULL,
                chase_dist_pct DOUBLE PRECISION NOT NULL
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """))

def read_meta(engine, key: str) -> str | None:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT value FROM meta WHERE key=:k"), {"k": key}).fetchone()
        return row[0] if row else None

def write_meta(engine, key: str, value: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO meta (key, value)
            VALUES (:k, :v)
            ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
        """), {"k": key, "v": value})

def save_plays(engine, plans: list[Plan]) -> None:
    computed_at = datetime.utcnow()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM plays WHERE computed_at < NOW() - INTERVAL '7 days'"))
        for idx, p in enumerate(plans, start=1):
            conn.execute(
                text("""
                    INSERT INTO plays (
                        computed_at, rank, ticker, direction, grade, score,
                        entry_now, ema20, stop, target, rr, confidence, vol_risk,
                        chase_label, chase_dist_pct
                    )
                    VALUES (
                        :computed_at, :rank, :ticker, :direction, :grade, :score,
                        :entry_now, :ema20, :stop, :target, :rr, :confidence, :vol_risk,
                        :chase_label, :chase_dist_pct
                    )
                """),
                {
                    "computed_at": computed_at,
                    "rank": idx,
                    "ticker": p.ticker,
                    "direction": p.direction,
                    "grade": p.grade,
                    "score": p.score,
                    "entry_now": p.entry_now,
                    "ema20": p.ema20,
                    "stop": p.stop,
                    "target": p.target,
                    "rr": p.rr,
                    "confidence": p.confidence,
                    "vol_risk": p.vol_risk,
                    "chase_label": p.chase_label,
                    "chase_dist_pct": p.chase_dist_pct
                }
            )

# =========================
# CHANGE DETECTION + ALERT
# =========================

def primary_key(plan: Plan | None) -> str:
    if plan is None:
        return "NONE"
    # Include the pieces that matter for decisions
    return f"{plan.ticker}|{plan.direction}|{plan.grade}"

def format_play(plan: Plan) -> str:
    return (
        f"**{plan.dot} {plan.ticker} (Grade {plan.grade})**\n"
        f"Direction: {plan.direction}\n"
        f"Entry now: {plan.entry_now:.2f}\n"
        f"Stop: {plan.stop:.2f}\n"
        f"Target: {plan.target:.2f}\n"
        f"R/R: {plan.rr:.2f} | Score: {plan.score:.1f}/10\n"
        f"Chase: {plan.chase_dot} {plan.chase_label} ({plan.chase_dist_pct:.2f}%)"
    )

def alert_if_changed(engine, primary: Plan | None, backups: list[Plan]) -> None:
    old_key = read_meta(engine, "last_primary_key") or "NONE"
    new_key = primary_key(primary)

    if new_key == old_key:
        return

    # Persist new key
    write_meta(engine, "last_primary_key", new_key)

    # Send alert
    header = f"📣 **Greenlight update** ({now_ct_str()})\n"
    if primary is None:
        msg = header + "No Grade A or B setups right now. No trade is a valid outcome."
        send_discord(msg)
        return

    lines = [header]
    lines.append("**PRIMARY PLAY changed**")
    lines.append(format_play(primary))

    if backups:
        lines.append("\n**BACKUPS**")
        for i, b in enumerate(backups[:2], start=1):
            lines.append(f"\nBackup {i}")
            lines.append(format_play(b))

    send_discord("\n".join(lines))

# =========================
# MAIN
# =========================

def main():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    engine = get_engine()
    ensure_tables(engine)

    plans: list[Plan] = []
    for t in UNIVERSE:
        p = build_plan_v1(t)
        if p is not None and p.grade in ("A", "B"):
            plans.append(p)

    plans.sort(key=lambda x: x.score, reverse=True)

    top3 = plans[:3]
    primary = top3[0] if len(top3) >= 1 else None
    backups = top3[1:3] if len(top3) > 1 else []

    # Save current plays to DB (even if empty)
    save_plays(engine, top3)

    # Heartbeat rule D: only alert on meaningful change (primary key change)
    alert_if_changed(engine, primary, backups)

    print(f"{datetime.utcnow().isoformat(timespec='seconds')} plans={len(plans)} top_saved={len(top3)} primary={primary_key(primary)}")

if __name__ == "__main__":
    main()