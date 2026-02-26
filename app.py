import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "Greenlight Decision Engine (V4)"
UNIVERSE = [
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV",
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
    "JPM", "BAC", "NFLX", "COIN"
]

RISK_MODES = {"Conservative": 10, "Balanced": 15, "Spicy": 20}

OPTION_STOP_PCT = 0.35
OPTION_TAKE_PCT = 0.60
OPTION_DTE_HINT = "30 to 45 DTE"
OPTION_STRIKE_HINT = "Near the money"

A_CONF_MIN, A_RR_MIN, A_VOL_MAX = 7.5, 2.0, 6.5
B_CONF_MIN, B_RR_MIN, B_VOL_MAX = 6.5, 1.6, 7.5

CHASE_GREEN_MAX = 0.5
CHASE_YELLOW_MAX = 1.5

AUTO_SIM_DAYS = 5

TRADES_CSV = "trades_log.csv"
SIGNALS_CSV = "signals_log.csv"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
      .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 16px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 14px;
      }
      .title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 4px;
      }
      .pill {
        display:inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        margin-right: 8px;
        margin-bottom: 6px;
      }
      .hr { height:1px; background: rgba(255,255,255,0.10); margin: 12px 0; }
      .muted { opacity: 0.78; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(APP_TITLE)
st.caption("Educational tool. Not financial advice. Built for clarity and discipline.")


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


@st.cache_data(ttl=60 * 5)
def fetch_intraday_5m(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60)
def fetch_daily_1y(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


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
    return "🔴", "Overextended, wait for pullback", dist_pct


def position_sizing(account: float, risk_dollars: float, entry: float, stop: float, per_trade_cap: float) -> dict:
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0 or entry <= 0:
        return {"shares": 0, "cost": 0.0, "risk": 0.0}
    risk_shares = math.floor(risk_dollars / per_unit_risk)
    cap_shares = math.floor(min(account, per_trade_cap) / entry)
    shares = max(0, min(risk_shares, cap_shares))
    cost = shares * entry
    risk = shares * per_unit_risk
    return {"shares": shares, "cost": float(cost), "risk": float(risk)}


def option_budget(risk_dollars: float, per_trade_cap: float) -> float:
    return float(min(risk_dollars * 3, per_trade_cap))


def safe_float(x, fallback=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(fallback)


def market_regime_spy() -> tuple[str, str]:
    df = fetch_daily_1y("SPY")
    if df is None or df.empty or len(df) < 210:
        return "🟡", "Mixed"
    close = df["Close"]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1]
    last = float(close.iloc[-1])

    above50 = last > float(ema50)
    above200 = last > float(ema200)

    if above50 and above200:
        return "🟢", "Trending"
    if (above50 and not above200) or (not above50 and above200):
        return "🟡", "Mixed"
    return "🔴", "Defensive"


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


def build_plan_intraday(ticker: str) -> Plan | None:
    df = fetch_intraday_5m(ticker)
    if df is None or df.empty or len(df) < 90:
        return None

    close = df["Close"]
    last = float(close.iloc[-1])

    ema20_s = close.ewm(span=20).mean()
    ema50_s = close.ewm(span=50).mean()
    ema20 = float(ema20_s.iloc[-1])
    ema50 = float(ema50_s.iloc[-1])

    r14 = float(rsi(close, 14).iloc[-1])

    a = atr(df, 14).iloc[-1]
    a = float(a) if not np.isnan(a) else last * 0.02

    atr_pct = a / last if last > 0 else np.nan
    vol_risk = float(np.clip((atr_pct * 100) / 2, 0, 10)) if not np.isnan(atr_pct) else 5.0

    bullish = (last > ema20) and (ema20 > ema50) and (r14 < 75)
    bearish = (last < ema20) and (ema20 < ema50) and (r14 > 25)
    if not bullish and not bearish:
        return None

    direction = "Bullish" if bullish else "Bearish"

    stop_dist = a * 1.2
    target_dist = stop_dist * 2.0
    stop = last - stop_dist if direction == "Bullish" else last + stop_dist
    target = last + target_dist if direction == "Bullish" else last - target_dist
    rr_val = abs(target - last) / max(abs(last - stop), 1e-9)

    ema20_prev = float(ema20_s.iloc[-20]) if len(ema20_s) >= 21 else ema20
    ema_slope = (ema20 - ema20_prev) / max(ema20_prev, 1e-9)
    momentum_score = float(np.clip((ema_slope * 100) * 2 + 5, 0, 10))
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


def compute_entry_zone(plan: Plan) -> tuple[float, float, float]:
    low = min(plan.ema20, plan.entry_now)
    high = max(plan.ema20, plan.entry_now)
    ideal = plan.ema20
    return float(low), float(high), float(ideal)


def load_csv(path: str, columns: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns]
    except Exception:
        return pd.DataFrame(columns=columns)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


TRADE_COLS = [
    "logged_at", "source", "ticker", "direction", "grade", "timeframe",
    "entry", "stop", "target",
    "risk_mode", "risk_dollars",
    "shares", "shares_risk_dollars", "shares_profit_at_target",
    "option_max_premium", "option_profit_at_take", "option_loss_at_stop",
    "result", "result_R", "notes"
]

SIGNAL_COLS = [
    "signal_id", "recorded_at", "ticker", "direction", "grade", "timeframe",
    "entry", "stop", "target", "risk_mode",
    "shares", "shares_risk_dollars", "shares_profit_at_target",
    "option_max_premium", "option_profit_at_take", "option_loss_at_stop",
    "status", "outcome", "outcome_R", "simulated_at", "sim_note"
]


def ensure_state():
    if "trades_df" not in st.session_state:
        st.session_state.trades_df = load_csv(TRADES_CSV, TRADE_COLS)
    if "signals_df" not in st.session_state:
        st.session_state.signals_df = load_csv(SIGNALS_CSV, SIGNAL_COLS)


ensure_state()

with st.sidebar:
    st.header("Controls")

    account = st.number_input("Account size ($)", min_value=100.0, value=1000.0, step=100.0)
    max_trades = st.slider("Max open trades", min_value=1, max_value=6, value=2, step=1)
    max_exposure_pct = st.slider("Max total exposure (%)", min_value=10, max_value=100, value=50, step=5)

    st.divider()
    risk_mode = st.selectbox("Risk mode", ["Conservative", "Balanced", "Spicy"], index=1)
    starter_mode = st.checkbox("Starter position mode (50% now, add later)", value=False)

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_sec = st.slider("Refresh seconds", min_value=30, max_value=600, value=120, step=30)

per_trade_cap = (account * (max_exposure_pct / 100.0)) / max_trades
risk_dollars_raw = RISK_MODES[risk_mode]

m_dot, m_state = market_regime_spy()
risk_scale = 0.75 if m_state == "Defensive" else 1.0
risk_dollars = float(risk_dollars_raw * risk_scale)

plans: list[Plan] = []
for t in UNIVERSE:
    p = build_plan_intraday(t)
    if p is not None and p.grade in ("A", "B"):
        plans.append(p)

plans.sort(key=lambda x: x.score, reverse=True)

tab_play, tab_list, tab_journal = st.tabs(["Plays", "Setups List", "Journal"])


def format_money(x: float) -> str:
    if x is None or np.isnan(x):
        return "-"
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def render_play_card(label: str, plan: Plan):
    entry_low, entry_high, ideal = compute_entry_zone(plan)

    shares_enabled = (plan.direction == "Bullish")
    sizing = position_sizing(account, risk_dollars, plan.entry_now, plan.stop, per_trade_cap) if shares_enabled else {"shares": 0, "cost": 0.0, "risk": 0.0}
    shares = int(sizing["shares"])
    shares_risk = float(sizing["risk"])
    shares_profit = float(shares * abs(plan.target - plan.entry_now))

    starter_shares_now = math.floor(shares * 0.5) if starter_mode else shares
    starter_shares_add = shares - starter_shares_now if starter_mode else 0

    options_allowed = (plan.grade == "A")
    opt_max = option_budget(risk_dollars, per_trade_cap) if options_allowed else 0.0
    opt_profit_take = opt_max * OPTION_TAKE_PCT
    opt_loss_stop = opt_max * OPTION_STOP_PCT

    if plan.direction == "Bullish":
        option_action = "BUY 1 CALL"
    else:
        option_action = "BUY 1 PUT"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="title">{label}: {plan.dot} {plan.ticker} (Grade {plan.grade})</div>', unsafe_allow_html=True)

    st.markdown(
        f'<span class="pill">{plan.direction}</span>'
        f'<span class="pill">{plan.timeframe}</span>'
        f'<span class="pill">Market: {m_dot} {m_state}</span>'
        f'<span class="pill">Chase: {plan.chase_dot} {plan.chase_label}</span>'
        f'<span class="pill">Score {plan.score:.1f}/10</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    left, right = st.columns([2, 2])

    with left:
        st.markdown("### The play")
        st.caption("Shares are the default. Options are optional and only shown on Grade A.")

        # Shares-first default
        if plan.direction == "Bullish":
            if shares > 0:
                if starter_mode:
                    st.markdown(f"**WHAT TO BUY (DEFAULT):** Buy **{starter_shares_now} shares** now")
                    if starter_shares_add > 0:
                        st.markdown(f"**ADD LATER:** Add **{starter_shares_add} shares** on a pullback near **EMA20 {ideal:.2f}** if it holds")
                else:
                    st.markdown(f"**WHAT TO BUY (DEFAULT):** Buy **{shares} shares**")
            else:
                st.markdown("**WHAT TO BUY (DEFAULT):** Shares size is 0 with current risk and cap settings. Increase risk mode or lower caps.")
        else:
            st.markdown("**WHAT TO BUY (DEFAULT):** Skip shares (we are not shorting in V4). Use options if Grade A.")

        # Options are explicitly optional
        if options_allowed:
            st.markdown(f"**OPTION (Optional leverage):** {option_action}")
            st.markdown(f"- Expiration: {OPTION_DTE_HINT}")
            st.markdown(f"- Strike: {OPTION_STRIKE_HINT}")
            st.markdown(f"- Max premium: **{format_money(opt_max)}**")
            st.markdown(f"- Take profit at +{int(OPTION_TAKE_PCT*100)}%: **about {format_money(opt_profit_take)} profit**")
            st.markdown(f"- Stop at -{int(OPTION_STOP_PCT*100)}%: **about {format_money(opt_loss_stop)} loss**")
        else:
            st.markdown("**OPTIONS:** Not shown because this is Grade B")

    with right:
        st.markdown("### Levels")
        st.markdown(f"**ENTRY ZONE:** {entry_low:.2f} to {entry_high:.2f}")
        st.markdown(f"**IDEAL PULLBACK:** near EMA20 {ideal:.2f}")
        st.markdown(f"**STOP:** {plan.stop:.2f}")
        st.markdown(f"**TARGET:** {plan.target:.2f}")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("### Money (based on your risk mode)")
        st.markdown(f"Risk mode: **{risk_mode}** (risk budget {format_money(risk_dollars_raw)}")
        if m_state == "Defensive":
            st.markdown(f"Market is Defensive, sizing reduced 25%, effective risk {format_money(risk_dollars)})")
        else:
            st.markdown(f"Effective risk {format_money(risk_dollars)})")

        if shares_enabled:
            st.markdown(f"Shares risk if stop hits: **{format_money(shares_risk)}**")
            st.markdown(f"Shares profit if target hits: **{format_money(shares_profit)}**")

    if plan.chase_dot == "🔴":
        st.error("Overextended. Do not chase. Wait for pullback toward EMA20, then reassess.")

    with st.expander("Details (optional)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry now", f"{plan.entry_now:.2f}")
        c2.metric("R/R", f"{plan.rr:.2f}")
        c3.metric("Confidence", f"{plan.confidence:.1f}/10")
        c4.metric("Vol risk", f"{plan.vol_risk:.1f}/10")
        st.caption(f"Chase distance from EMA20: {plan.chase_dist_pct:.2f}%")

    a1, a2, a3, a4 = st.columns([1.2, 1.2, 1.6, 2.0])

    def build_trade_row(result: str, source: str) -> dict:
        return {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "ticker": plan.ticker,
            "direction": plan.direction,
            "grade": plan.grade,
            "timeframe": plan.timeframe,
            "entry": float(plan.entry_now),
            "stop": float(plan.stop),
            "target": float(plan.target),
            "risk_mode": risk_mode,
            "risk_dollars": float(risk_dollars),
            "shares": int(shares),
            "shares_risk_dollars": float(shares_risk),
            "shares_profit_at_target": float(shares_profit),
            "option_max_premium": float(opt_max),
            "option_profit_at_take": float(opt_profit_take),
            "option_loss_at_stop": float(opt_loss_stop),
            "result": result,
            "result_R": np.nan,
            "notes": ""
        }

    with a1:
        if st.button(f"Log WIN ({label})", key=f"win_{label}_{plan.ticker}"):
            row = build_trade_row("WIN", "manual")
            st.session_state.trades_df = pd.concat([st.session_state.trades_df, pd.DataFrame([row])], ignore_index=True)
            save_csv(st.session_state.trades_df, TRADES_CSV)
            st.success("Logged win.")
    with a2:
        if st.button(f"Log LOSS ({label})", key=f"loss_{label}_{plan.ticker}"):
            row = build_trade_row("LOSS", "manual")
            st.session_state.trades_df = pd.concat([st.session_state.trades_df, pd.DataFrame([row])], ignore_index=True)
            save_csv(st.session_state.trades_df, TRADES_CSV)
            st.success("Logged loss.")
    with a3:
        if st.button(f"Record signal ({label})", key=f"record_{label}_{plan.ticker}"):
            signal_id = f"{plan.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            row = {
                "signal_id": signal_id,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "ticker": plan.ticker,
                "direction": plan.direction,
                "grade": plan.grade,
                "timeframe": plan.timeframe,
                "entry": float(plan.entry_now),
                "stop": float(plan.stop),
                "target": float(plan.target),
                "risk_mode": risk_mode,
                "shares": int(shares),
                "shares_risk_dollars": float(shares_risk),
                "shares_profit_at_target": float(shares_profit),
                "option_max_premium": float(opt_max),
                "option_profit_at_take": float(opt_profit_take),
                "option_loss_at_stop": float(opt_loss_stop),
                "status": "open",
                "outcome": "",
                "outcome_R": np.nan,
                "simulated_at": "",
                "sim_note": ""
            }
            st.session_state.signals_df = pd.concat([st.session_state.signals_df, pd.DataFrame([row])], ignore_index=True)
            save_csv(st.session_state.signals_df, SIGNALS_CSV)
            st.success("Signal recorded for auto-sim.")
    with a4:
        st.caption("Manual logs are for real trades. Signal record is for auto-sim validation.")

    st.markdown("</div>", unsafe_allow_html=True)


def build_setups_table(plans_list: list[Plan]) -> pd.DataFrame:
    rows = []
    for p in plans_list:
        entry_low, entry_high, ideal = compute_entry_zone(p)
        shares_enabled = (p.direction == "Bullish")
        sizing = position_sizing(account, risk_dollars, p.entry_now, p.stop, per_trade_cap) if shares_enabled else {"shares": 0, "cost": 0.0, "risk": 0.0}
        shares = int(sizing["shares"])
        shares_risk = float(sizing["risk"])
        shares_profit = float(shares * abs(p.target - p.entry_now))

        options_allowed = (p.grade == "A")
        opt_max = option_budget(risk_dollars, per_trade_cap) if options_allowed else 0.0
        opt_profit_take = opt_max * OPTION_TAKE_PCT
        opt_loss_stop = opt_max * OPTION_STOP_PCT

        rows.append({
            "": p.dot,
            "Ticker": p.ticker,
            "Grade": p.grade,
            "Dir": p.direction,
            "Market": f"{m_dot} {m_state}",
            "Chase": f"{p.chase_dot} {p.chase_label}",
            "Entry Zone": f"{entry_low:.2f} to {entry_high:.2f}",
            "Stop": round(p.stop, 2),
            "Target": round(p.target, 2),
            "Shares": shares,
            "Shares Risk $": round(shares_risk, 2),
            "Shares Profit $": round(shares_profit, 2),
            "Options?": "Yes" if options_allowed else "No",
            "Opt Max $": round(opt_max, 2),
            "Opt Profit $": round(opt_profit_take, 2),
            "Opt Loss $": round(opt_loss_stop, 2),
            "Score": round(p.score, 1),
        })
    return pd.DataFrame(rows)


def next_trading_days_prices(ticker: str, start_dt: date, days: int) -> pd.DataFrame:
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = (start_dt + timedelta(days=days + 10)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_str, end=end_str, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.head(days)


def simulate_signal_outcome(row: pd.Series) -> tuple[str, float, str]:
    ticker = str(row["ticker"])
    direction = str(row["direction"])
    entry = safe_float(row["entry"])
    stop = safe_float(row["stop"])
    target = safe_float(row["target"])

    try:
        recorded = datetime.fromisoformat(str(row["recorded_at"]))
        start_day = (recorded.date() + timedelta(days=1))
    except Exception:
        start_day = (date.today() + timedelta(days=1))

    df = next_trading_days_prices(ticker, start_day, AUTO_SIM_DAYS)
    if df.empty:
        return "NO_DATA", np.nan, "No daily data returned for sim window."

    R = abs(entry - stop)
    if R <= 0:
        return "NO_DATA", np.nan, "Invalid stop distance."

    for _, day in df.iterrows():
        high = safe_float(day.get("High", np.nan))
        low = safe_float(day.get("Low", np.nan))

        if direction == "Bullish":
            hit_stop = (low <= stop)
            hit_target = (high >= target)
        else:
            hit_stop = (high >= stop)
            hit_target = (low <= target)

        if hit_stop and hit_target:
            return "LOSS", -1.0, "Both stop and target hit same day, counted as loss (conservative)."
        if hit_stop:
            return "LOSS", -1.0, "Stop hit first."
        if hit_target:
            return "WIN", abs(target - entry) / R, "Target hit first."

    return "NO_DATA", np.nan, f"No stop or target hit in {AUTO_SIM_DAYS} trading days."


with tab_play:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">At a glance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="pill">Market: {m_dot} {m_state}</span>'
        f'<span class="pill">Risk mode: {risk_mode}</span>'
        f'<span class="pill">Per-trade cap: {format_money(per_trade_cap)}</span>'
        f'<span class="pill">Effective risk budget: {format_money(risk_dollars)}</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='muted'>Primary play is the best setup right now. Backups are next best. "
        "Shares are the default action.</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not plans:
        st.warning("No Grade A or B setups right now. No trade is a valid outcome.")
    else:
        primary = plans[0]
        backups = plans[1:3] if len(plans) > 1 else []
        render_play_card("PRIMARY PLAY", primary)
        for i, b in enumerate(backups, start=1):
            render_play_card(f"BACKUP {i}", b)

    if auto_refresh:
        st.caption(f"Auto-refreshing every {refresh_sec} seconds.")
        time.sleep(refresh_sec)
        st.rerun()

with tab_list:
    st.subheader("A and B setups only, ranked")
    st.caption("List view. Plays tab is the one you act on.")
    if not plans:
        st.warning("No setups right now.")
    else:
        df = build_setups_table(plans)
        st.dataframe(df, width="stretch", hide_index=True)

with tab_journal:
    st.subheader("Journal and validation")

    trades = st.session_state.trades_df.copy()
    if not trades.empty:

        def infer_R(row):
            entry = safe_float(row["entry"])
            stop = safe_float(row["stop"])
            target = safe_float(row["target"])
            if np.isnan(entry) or np.isnan(stop) or np.isnan(target):
                return np.nan
            R = abs(entry - stop)
            if R <= 0:
                return np.nan
            if str(row["result"]).upper() == "WIN":
                return abs(target - entry) / R
            if str(row["result"]).upper() == "LOSS":
                return -1.0
            return np.nan

        if "result_R" in trades.columns:
            trades["result_R"] = trades["result_R"].fillna(trades.apply(infer_R, axis=1))
        else:
            trades["result_R"] = trades.apply(infer_R, axis=1)

        wins = (trades["result"].astype(str).str.upper() == "WIN").sum()
        losses = (trades["result"].astype(str).str.upper() == "LOSS").sum()
        total = wins + losses
        win_rate = (wins / total) * 100 if total > 0 else 0.0
        total_R = safe_float(trades["result_R"].sum(), 0.0)
        avg_R = safe_float(trades["result_R"].mean(), np.nan)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="title">Edge tracker</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades logged", f"{total}")
        c2.metric("Win rate", f"{win_rate:.1f}%")
        c3.metric("Avg R", "-" if np.isnan(avg_R) else f"{avg_R:.2f}R")
        c4.metric("Total R", f"{total_R:.2f}R")
        st.markdown(
            "<div class='muted'>Goal: get to 50 trades with positive Total R, while respecting entry discipline.</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No trades logged yet. Use the Log WIN or Log LOSS buttons on the Plays tab.")

    st.markdown("### Trade log (manual)")
    st.dataframe(st.session_state.trades_df.tail(100), width="stretch", hide_index=True)

    st.markdown("### Signals log (for auto-sim)")
    signals = st.session_state.signals_df.copy()
    st.dataframe(signals.tail(100), width="stretch", hide_index=True)

    colA, colB, colC = st.columns([1.2, 1.6, 2.2])

    with colA:
        if st.button("Auto-sim open signals"):
            if signals.empty:
                st.warning("No signals recorded.")
            else:
                updated = signals.copy()
                changed = 0
                for i in range(len(updated)):
                    if str(updated.loc[i, "status"]) != "open":
                        continue
                    outcome, outR, note = simulate_signal_outcome(updated.loc[i])
                    if outcome in ("WIN", "LOSS"):
                        updated.loc[i, "status"] = "closed"
                        updated.loc[i, "outcome"] = outcome
                        updated.loc[i, "outcome_R"] = outR
                        updated.loc[i, "simulated_at"] = datetime.now().isoformat(timespec="seconds")
                        updated.loc[i, "sim_note"] = note
                        changed += 1
                st.session_state.signals_df = updated
                save_csv(st.session_state.signals_df, SIGNALS_CSV)
                st.success(f"Auto-sim completed. Updated {changed} signals.")

    with colB:
        if st.button("Promote sim results to trade log"):
            sig = st.session_state.signals_df.copy()
            done = 0
            if sig.empty:
                st.warning("No signals to promote.")
            else:
                to_add = []
                for _, r in sig.iterrows():
                    if str(r.get("status", "")) != "closed":
                        continue
                    outcome = str(r.get("outcome", "")).upper()
                    if outcome not in ("WIN", "LOSS"):
                        continue
                    trade_row = {
                        "logged_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "auto-sim",
                        "ticker": r["ticker"],
                        "direction": r["direction"],
                        "grade": r["grade"],
                        "timeframe": r["timeframe"],
                        "entry": safe_float(r["entry"]),
                        "stop": safe_float(r["stop"]),
                        "target": safe_float(r["target"]),
                        "risk_mode": r.get("risk_mode", ""),
                        "risk_dollars": np.nan,
                        "shares": int(safe_float(r.get("shares", 0), 0)),
                        "shares_risk_dollars": safe_float(r.get("shares_risk_dollars", np.nan)),
                        "shares_profit_at_target": safe_float(r.get("shares_profit_at_target", np.nan)),
                        "option_max_premium": safe_float(r.get("option_max_premium", np.nan)),
                        "option_profit_at_take": safe_float(r.get("option_profit_at_take", np.nan)),
                        "option_loss_at_stop": safe_float(r.get("option_loss_at_stop", np.nan)),
                        "result": outcome,
                        "result_R": safe_float(r.get("outcome_R", np.nan)),
                        "notes": f"Auto-sim: {r.get('sim_note','')}"
                    }
                    to_add.append(trade_row)
                if to_add:
                    st.session_state.trades_df = pd.concat([st.session_state.trades_df, pd.DataFrame(to_add)], ignore_index=True)
                    save_csv(st.session_state.trades_df, TRADES_CSV)
                    done = len(to_add)
                st.success(f"Promoted {done} simulated trades into the trade log.")

    with colC:
        st.caption(
            "Auto-sim rules: looks ahead 5 trading days using daily bars. "
            "If stop and target hit the same day, it counts as a loss (conservative)."
        )