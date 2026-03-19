"""
Database persistence layer for the trading bot.
Uses PostgreSQL when DATABASE_URL is set (Render), falls back to SQLite locally.
Stores closed trades and engine state so they survive server restarts/deploys.
"""

import os
import sqlite3
import threading
from datetime import datetime

# Try to import psycopg2 for PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trades.db")
_lock = threading.Lock()
_use_pg = False
_pg_url = None


# ── Connection Helpers ──────────────────────────────────────────────────────

def _get_pg_conn():
    """Get a PostgreSQL connection."""
    conn = psycopg2.connect(_pg_url)
    conn.autocommit = False
    return conn


def _get_sqlite_conn() -> sqlite3.Connection:
    """Get a SQLite connection."""
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_conn():
    """Get the appropriate database connection."""
    if _use_pg:
        return _get_pg_conn()
    return _get_sqlite_conn()


# ── Init ────────────────────────────────────────────────────────────────────

def init_db(path: str | None = None):
    """Create tables if they don't exist. Call once on startup."""
    global _DB_PATH, _use_pg, _pg_url

    # Check for PostgreSQL URL
    _pg_url = os.environ.get("DATABASE_URL", "").strip()
    if _pg_url and _HAS_PG:
        # Render uses postgres:// but psycopg2 needs postgresql://
        if _pg_url.startswith("postgres://"):
            _pg_url = _pg_url.replace("postgres://", "postgresql://", 1)
        _use_pg = True
        print(f"[DB] Using PostgreSQL (Render persistent database)")
        _init_pg()
    else:
        _use_pg = False
        if path:
            _DB_PATH = path
        print(f"[DB] Using SQLite at {_DB_PATH}")
        _init_sqlite()


def _init_pg():
    """Create PostgreSQL tables."""
    with _lock:
        conn = _get_pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    pnl DOUBLE PRECISION NOT NULL,
                    reason_open TEXT,
                    reason_close TEXT,
                    opened_at TEXT,
                    closed_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value DOUBLE PRECISION NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()


def _init_sqlite():
    """Create SQLite tables."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with _lock:
        conn = _get_sqlite_conn()
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


# ── Trade Operations ────────────────────────────────────────────────────────

def save_trade(trade) -> None:
    """Insert a closed trade into the database."""
    with _lock:
        conn = _get_conn()
        try:
            if _use_pg:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO closed_trades
                       (id, symbol, side, entry_price, exit_price, quantity, pnl,
                        reason_open, reason_close, opened_at, closed_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO UPDATE SET
                        exit_price = EXCLUDED.exit_price,
                        pnl = EXCLUDED.pnl,
                        reason_close = EXCLUDED.reason_close,
                        closed_at = EXCLUDED.closed_at""",
                    (
                        trade.id, trade.symbol, trade.side,
                        trade.entry_price, trade.exit_price, trade.quantity, trade.pnl,
                        trade.reason_open, trade.reason_close,
                        trade.opened_at, trade.closed_at,
                    ),
                )
                conn.commit()
            else:
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
            if _use_pg:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT * FROM closed_trades ORDER BY closed_at ASC")
                return [dict(row) for row in cur.fetchall()]
            else:
                rows = conn.execute(
                    "SELECT * FROM closed_trades ORDER BY closed_at ASC"
                ).fetchall()
                return [dict(row) for row in rows]
        finally:
            conn.close()


# ── State Operations ────────────────────────────────────────────────────────

def save_state(state: dict) -> None:
    """Save engine state (balance, stats) as key-value pairs."""
    now = datetime.now().isoformat()
    with _lock:
        conn = _get_conn()
        try:
            if _use_pg:
                cur = conn.cursor()
                for key, value in state.items():
                    cur.execute(
                        """INSERT INTO engine_state (key, value, updated_at)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (key) DO UPDATE SET
                            value = EXCLUDED.value,
                            updated_at = EXCLUDED.updated_at""",
                        (key, float(value), now),
                    )
                conn.commit()
            else:
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
            if _use_pg:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT key, value FROM engine_state")
                rows = cur.fetchall()
                if not rows:
                    return None
                return {row["key"]: row["value"] for row in rows}
            else:
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
            if _use_pg:
                cur = conn.cursor()
                cur.execute("DELETE FROM closed_trades")
                cur.execute("DELETE FROM engine_state")
                conn.commit()
            else:
                conn.executescript("""
                    DELETE FROM closed_trades;
                    DELETE FROM engine_state;
                """)
                conn.commit()
        finally:
            conn.close()
