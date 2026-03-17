"""
SQLite persistence layer for the trading bot.
Stores closed trades and engine state so they survive server restarts.
Uses Python's built-in sqlite3 — zero new dependencies.
"""

import os
import sqlite3
import threading
from datetime import datetime


_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trades.db")
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read/write
    return conn


def init_db(path: str | None = None):
    """Create tables if they don't exist. Call once on startup."""
    global _DB_PATH
    if path:
        _DB_PATH = path

    # Ensure data directory exists
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)

    with _lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL NOT NULL,
                    reason_open TEXT,
                    reason_close TEXT,
                    opened_at TEXT,
                    closed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()


def save_trade(trade) -> None:
    """Insert a closed trade into the database."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO closed_trades
                   (id, symbol, side, entry_price, exit_price, quantity, pnl,
                    reason_open, reason_close, opened_at, closed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.id, trade.symbol, trade.side,
                    trade.entry_price, trade.exit_price, trade.quantity, trade.pnl,
                    trade.reason_open, trade.reason_close,
                    trade.opened_at, trade.closed_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def load_trades() -> list[dict]:
    """Load all closed trades from the database, ordered by closed_at."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM closed_trades ORDER BY closed_at ASC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def save_state(state: dict) -> None:
    """Save engine state (balance, stats) as key-value pairs."""
    now = datetime.now().isoformat()
    with _lock:
        conn = _get_conn()
        try:
            for key, value in state.items():
                conn.execute(
                    """INSERT OR REPLACE INTO engine_state (key, value, updated_at)
                       VALUES (?, ?, ?)""",
                    (key, float(value), now),
                )
            conn.commit()
        finally:
            conn.close()


def load_state() -> dict | None:
    """Load saved engine state. Returns dict or None if no state saved."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute("SELECT key, value FROM engine_state").fetchall()
            if not rows:
                return None
            return {row["key"]: row["value"] for row in rows}
        finally:
            conn.close()


def reset_db() -> None:
    """Wipe all data — fresh start."""
    with _lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                DELETE FROM closed_trades;
                DELETE FROM engine_state;
            """)
            conn.commit()
        finally:
            conn.close()
