import os
import requests
import pytz
from datetime import datetime
from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "NVDA",
    "AAPL", "MSFT", "META", "AMZN", "TSLA"
]

# =========================
# DB SETUP
# =========================

engine = create_engine(DATABASE_URL)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                ticker TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

# =========================
# DISCORD
# =========================

def send_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        print("No Discord webhook set.")
        return
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": msg},
            timeout=10
        )
    except Exception as e:
        print("Discord error:", e)

# =========================
# MONITOR LOGIC
# =========================

def check_signals():
    # Placeholder logic
    promoted = 0
    closed = 0

    # Example structure for future logic
    # for ticker in TICKERS:
    #     evaluate signal
    #     update promoted / closed

    return promoted, closed

# =========================
# HEARTBEAT
# =========================

def send_heartbeat(promoted, closed):
    try:
        central = pytz.timezone("US/Central")
        now_ct = datetime.now(central).strftime("%Y-%m-%d %I:%M %p CT")
    except Exception:
        now_ct = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    heartbeat = (
        "🟢 **Greenlight Monitor**\n"
        f"Checked tickers: {len(TICKERS)}\n"
        f"Promoted: {promoted}\n"
        f"Closed: {closed}\n"
        f"Time: {now_ct}"
    )

    send_discord(heartbeat)

# =========================
# MAIN
# =========================

def main():
    if not DATABASE_URL:
        print("DATABASE_URL not set.")
        return

    init_db()

    promoted, closed = check_signals()

    print(f"{datetime.utcnow().isoformat()} closed={closed} promoted={promoted}")

    send_heartbeat(promoted, closed)

if __name__ == "__main__":
    main()