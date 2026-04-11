"""
Automated Multi-Asset Trading Engine
Trades: Commodities (BRENT, WTI, NGAS, COPPER, XAGUSD, XAUUSD, PLATINUM, PALLADIUM)
        Forex (EURUSD, GBPUSD, USDJPY) | Indices (SP500, NASDAQ, DOW)
Starting balance: $25,000 | Hard floor: $24,000
Strategy: Multi-indicator momentum + mean reversion with strict risk management
"""

import threading
import time
import json
import uuid
import os
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque

import yfinance as yf

try:
    import requests as _requests
except ImportError:
    _requests = None

from ai_brain import AIBrain
import db


# ── Symbol Mapping ──────────────────────────────────────────────────────────
# asset_class determines session multipliers and correlation grouping
SYMBOLS = {
    # ── Oil ──  (kept BRENT only — WTI was -$254, 31% WR)
    "BRENT":    {"yf": "BZ=F",      "name": "Brent Crude Oil",     "pip_value": 0.01,  "lot_size": 10,  "asset_class": "oil"},
    # ── Metals ──  (cut XAGUSD -$261, PALLADIUM -$77)
    "XAUUSD":   {"yf": "GC=F",      "name": "Gold (XAU/USD)",      "pip_value": 0.01,  "lot_size": 1,   "asset_class": "metals"},
    "PLATINUM": {"yf": "PL=F",      "name": "Platinum",            "pip_value": 0.01,  "lot_size": 1,   "asset_class": "metals"},
    "COPPER":   {"yf": "HG=F",      "name": "Copper",              "pip_value": 0.0001,"lot_size": 100, "asset_class": "metals"},
    # ── Forex ──  (cut GBPUSD 35% WR, -$39)
    "EURUSD":   {"yf": "EURUSD=X",  "name": "EUR/USD",             "pip_value": 0.0001,"lot_size": 1000,"asset_class": "forex"},
    "USDJPY":   {"yf": "USDJPY=X",  "name": "USD/JPY",             "pip_value": 0.01,  "lot_size": 1000,"asset_class": "forex", "pnl_ccy": "JPY"},
    # ── Indices ──  (cut NASDAQ -$225; kept SP500 +$62 and DOW for high WR)
    "SP500":    {"yf": "ES=F",      "name": "S&P 500 Futures",     "pip_value": 0.25,  "lot_size": 1,   "asset_class": "indices"},
    "DOW":      {"yf": "YM=F",      "name": "Dow Jones Futures",   "pip_value": 1.0,   "lot_size": 1,   "asset_class": "indices"},
}

# ── Configuration ───────────────────────────────────────────────────────────
STARTING_BALANCE = 25000.0
HARD_FLOOR = 24000.0
MAX_RISK_PER_TRADE_PCT = 0.20      # 0.20% per trade — slightly larger since fewer, higher-quality trades
MAX_OPEN_POSITIONS = 4             # reduced from 6 — focus on fewer, better positions
MAX_POSITIONS_PER_SYMBOL = 1
PRICE_FETCH_INTERVAL = 15          # seconds between price updates
STRATEGY_INTERVAL = 60             # seconds between strategy evaluations (was 20 — less churning)
MAX_DAILY_LOSS = 400.0             # tighter daily loss cap (was 600)
RISK_REWARD_RATIO = 1.5            # 1.5:1 R:R with wider stops = larger TP targets
SIGNAL_THRESHOLD = 3.0             # minimum composite score (was 2.0 — require strong conviction)
ATR_STOP_MULT = 3.0               # stop loss distance = 3x ATR (was 2x — give room to breathe)
ATR_TRAIL_TRIGGER = 2.5           # start trailing after 2.5x ATR move (was 2.0)
ATR_TRAIL_DIST = 1.5              # trail at 1.5x ATR behind price (was 1.2 — wider trail)
LOSS_COOLDOWN_SEC = 600            # 10 min cooldown per symbol after a losing trade (was 5 min)
TRADE_COOLDOWN_SEC = 1800          # 30 min cooldown per symbol after ANY trade (win or lose)
MAX_POSITIONS_PER_CLASS = 2        # max 2 positions in same asset class at once

# ── Correlation Groups ─────────────────────────────────────────────────────
# Block same-direction trades on highly correlated pairs
CORRELATED_PAIRS = [
    {"XAUUSD", "PLATINUM"},     # precious metals correlate
    {"SP500", "DOW"},           # US equity indices move together
]

# ── Trading Sessions (UTC hours) ──────────────────────────────────────────
# Each session defines position size multipliers per asset class and signal boost
# Multipliers: 0.0 = blocked, 0.5 = half size, 1.0 = full, 1.2 = boosted
SESSIONS = {
    "ASIAN": {
        "hours": (0, 8),
        "label": "Asian (Tokyo/Sydney)",
        "boost": 0.0,
        "class_mult": {
            "oil": 0.5, "energy": 0.5, "metals": 1.0,
            "forex": 0.7, "indices": 0.5,
        },
    },
    "LONDON": {
        "hours": (8, 13),
        "label": "London",
        "boost": 0.0,
        "class_mult": {
            "oil": 1.0, "energy": 1.0, "metals": 1.0,
            "forex": 1.0, "indices": 0.8,
        },
    },
    "OVERLAP": {
        "hours": (13, 16),
        "label": "London/NY Overlap (Peak)",
        "boost": 0.5,
        "class_mult": {
            "oil": 1.2, "energy": 1.2, "metals": 1.2,
            "forex": 1.2, "indices": 1.2,
        },
    },
    "NY": {
        "hours": (16, 21),
        "label": "New York",
        "boost": 0.0,
        "class_mult": {
            "oil": 1.0, "energy": 1.0, "metals": 0.8,
            "forex": 1.0, "indices": 1.0,
        },
    },
    "OFF_HOURS": {
        "hours": (21, 24),
        "label": "Off-Hours (Closed)",
        "boost": 0.0,
        "class_mult": {
            "oil": 0.0, "energy": 0.0, "metals": 0.0,
            "forex": 0.3, "indices": 0.0,  # forex still trades but reduced
        },
    },
}

# Asset class sets (derived from SYMBOLS for quick lookup)
OIL_SYMBOLS = {s for s, i in SYMBOLS.items() if i["asset_class"] == "oil"}
METALS_SYMBOLS = {s for s, i in SYMBOLS.items() if i["asset_class"] == "metals"}
ENERGY_SYMBOLS = {s for s, i in SYMBOLS.items() if i["asset_class"] == "energy"}
FOREX_SYMBOLS = {s for s, i in SYMBOLS.items() if i["asset_class"] == "forex"}
INDICES_SYMBOLS = {s for s, i in SYMBOLS.items() if i["asset_class"] == "indices"}


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    id: str
    symbol: str
    side: Side
    entry_price: float
    quantity: float          # in units (barrels, ounces)
    stop_loss: float
    take_profit: float
    reason: str
    opened_at: str
    unrealized_pnl: float = 0.0
    original_quantity: float = 0.0   # set on open, tracks initial size
    partial_closed: bool = False     # True after 50% taken at 1.5x ATR

    def to_dict(self):
        d = asdict(self)
        d["side"] = self.side.value
        return d


@dataclass
class ClosedTrade:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    reason_open: str
    reason_close: str
    opened_at: str
    closed_at: str

    def to_dict(self):
        return asdict(self)


class TradingEngine:
    def __init__(self):
        self.balance = STARTING_BALANCE
        self.equity = STARTING_BALANCE
        self.starting_balance = STARTING_BALANCE
        self.hard_floor = HARD_FLOOR

        self.positions: dict[str, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.trade_log: list[dict] = []       # human-readable log entries
        self.prices: dict[str, float] = {}
        self.price_history: dict[str, deque] = {
            sym: deque(maxlen=100) for sym in SYMBOLS
        }
        self.indicators: dict[str, dict] = {}
        self.htf_indicators: dict[str, dict] = {}  # 1-hour timeframe indicators

        self.running = False
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_equity = STARTING_BALANCE
        self.max_drawdown = 0.0
        self.status = "INITIALIZING"
        self._lock = threading.Lock()

        self._save_counter = 0  # for periodic state saves
        self._loss_cooldowns: dict[str, float] = {}  # {symbol: timestamp} — cooldown after losses
        self._trade_cooldowns: dict[str, float] = {}  # {symbol: timestamp} — cooldown after ANY trade
        self._recent_trades_pnl: deque = deque(maxlen=20)  # rolling 20-trade P&L for equity curve trading
        self._consecutive_losses: int = 0           # consecutive loss breaker
        self._loss_breaker_until: float = 0         # timestamp when breaker expires
        self._price_timestamps: dict[str, float] = {}  # {symbol: last_update_timestamp}
        self._atr_history: dict[str, deque] = {     # rolling ATR values for vol-norm sizing
            sym: deque(maxlen=50) for sym in SYMBOLS
        }

        # Adaptive per-symbol parameters (populated by walk-forward optimization)
        self.symbol_params: dict[str, dict] = {}  # {symbol: {signal_threshold, atr_stop_mult, risk_reward, max_risk_pct}}
        self._last_optimization_run: float = 0

        # AI Brain
        self.ai_brain = AIBrain(log_callback=lambda msg, level="INFO": self._log(msg, level=level))
        self.news_sentiment: dict = {}   # {symbol: {"score": float, "summary": str}}
        self.ai_explanations: dict = {}  # {position_id: str}

        # Database: init and restore
        db.init_db()
        self._restore_from_db()

    def _restore_from_db(self):
        """Restore balance, stats, and closed trades from SQLite on startup."""
        try:
            # Restore engine state
            saved = db.load_state()
            if saved:
                self.balance = saved.get("balance", STARTING_BALANCE)
                self.equity = saved.get("equity", self.balance)
                self.daily_pnl = saved.get("daily_pnl", 0.0)
                self.total_trades = int(saved.get("total_trades", 0))
                self.winning_trades = int(saved.get("winning_trades", 0))
                self.peak_equity = saved.get("peak_equity", self.balance)
                self.max_drawdown = saved.get("max_drawdown", 0.0)
                print(f"[DB] Restored state: balance=${self.balance:.2f}, {self.total_trades} trades")
            else:
                print("[DB] No saved state found — fresh start at $25,000")

            # Restore closed trades
            trade_rows = db.load_trades()
            for row in trade_rows:
                ct = ClosedTrade(
                    id=row["id"],
                    symbol=row["symbol"],
                    side=row["side"],
                    entry_price=row["entry_price"],
                    exit_price=row["exit_price"],
                    quantity=row["quantity"],
                    pnl=row["pnl"],
                    reason_open=row.get("reason_open", ""),
                    reason_close=row.get("reason_close", ""),
                    opened_at=row.get("opened_at", ""),
                    closed_at=row.get("closed_at", ""),
                )
                self.closed_trades.append(ct)

            if trade_rows:
                print(f"[DB] Restored {len(trade_rows)} closed trades from database")

        except Exception as e:
            print(f"[DB] Restore error (starting fresh): {e}")

    def _save_state_to_db(self):
        """Persist current engine state to SQLite."""
        try:
            db.save_state({
                "balance": self.balance,
                "equity": self.equity,
                "daily_pnl": self.daily_pnl,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "peak_equity": self.peak_equity,
                "max_drawdown": self.max_drawdown,
            })
        except Exception as e:
            print(f"[DB] Save state error: {e}")

    # ── Higher Timeframe (1H) Data ─────────────────────────────────────────

    def fetch_htf_data(self):
        """Fetch 1-hour candle data for all symbols and compute HTF indicators."""
        import pandas as pd
        for sym, info in SYMBOLS.items():
            try:
                hist = self._fetch_htf_candles(sym, info)
                if hist is not None and len(hist) >= 20:
                    self._compute_htf_indicators(sym, hist)
                    self._log(f"[HTF] {sym} 1H updated: {self.htf_indicators[sym].get('trend', '?')}, RSI={self.htf_indicators[sym].get('rsi', 0):.1f}")
            except Exception as e:
                self._log(f"[HTF] Error fetching {sym}: {e}", level="WARN")

    def _fetch_htf_candles(self, sym: str, info: dict):
        """Fetch 1-hour candles. Uses yfinance locally, query1 chart API on cloud."""
        import pandas as pd
        yf_sym = info["yf"]

        # Try yfinance first (local only)
        if not self._is_cloud:
            try:
                ticker = yf.Ticker(yf_sym)
                hist = ticker.history(period="1mo", interval="1h")
                if not hist.empty and len(hist) >= 20:
                    return hist
            except Exception:
                pass

        # Cloud-friendly: query1 chart API (no crumb needed)
        if _requests:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
                params = {"range": "1mo", "interval": "1h"}
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json",
                }
                resp = _requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
                    if quotes:
                        closes = [c for c in (quotes.get("close") or []) if c is not None]
                        highs = [h for h in (quotes.get("high") or []) if h is not None]
                        lows = [lo for lo in (quotes.get("low") or []) if lo is not None]
                        min_len = min(len(closes), len(highs), len(lows))
                        if min_len >= 20:
                            return pd.DataFrame({
                                "Close": closes[:min_len],
                                "High": highs[:min_len],
                                "Low": lows[:min_len],
                            })
            except Exception:
                pass

        return None

    def _compute_htf_indicators(self, symbol: str, hist):
        """Compute trend indicators from 1-hour candles."""
        closes = hist["Close"].values
        if len(closes) < 20:
            return

        sma_10 = float(closes[-10:].mean())
        sma_20 = float(closes[-20:].mean())
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        rsi = self._rsi(closes, 14)
        macd_line = ema_12 - ema_26
        macd_signal = macd_line * 0.8  # approximate
        if len(closes) >= 35:
            macd_series = []
            for i in range(34, len(closes)):
                e12 = self._ema(closes[:i+1], 12)
                e26 = self._ema(closes[:i+1], 26)
                macd_series.append(e12 - e26)
            if len(macd_series) >= 9:
                import numpy as np
                macd_signal = self._ema(np.array(macd_series), 9)
        macd_histogram = macd_line - macd_signal

        # Trend determination: use SMA crossover + MACD alignment
        trend = "NEUTRAL"
        if sma_10 > sma_20 and macd_histogram > 0:
            trend = "BULLISH"
        elif sma_10 > sma_20:
            trend = "LEAN_BULLISH"
        elif sma_10 < sma_20 and macd_histogram < 0:
            trend = "BEARISH"
        elif sma_10 < sma_20:
            trend = "LEAN_BEARISH"

        self.htf_indicators[symbol] = {
            "sma_10": sma_10,
            "sma_20": sma_20,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
            "trend": trend,
            "price": float(closes[-1]),
        }

    # ── Price Fetching (multi-source with cloud fallback) ───────────────────

    _is_cloud = os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL")

    def _run_with_timeout(self, fn, args, timeout_sec=20):
        """Run a function with a hard thread-based timeout."""
        result = [None, None]
        error = [None]
        def _target():
            try:
                result[0], result[1] = fn(*args)
            except Exception as e:
                error[0] = e
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            raise TimeoutError(f"Source timed out after {timeout_sec}s")
        if error[0]:
            raise error[0]
        return result[0], result[1]

    def _fetch_yfinance(self, sym: str, info: dict):
        """Primary source: Yahoo Finance via yfinance (skipped on cloud — hangs)."""
        if self._is_cloud:
            return None, None
        ticker = yf.Ticker(info["yf"])
        hist = ticker.history(period="2d", interval="5m")
        if hist.empty:
            hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return None, None
        price = float(hist["Close"].iloc[-1])
        return price, hist

    def _fetch_yahoo_download(self, sym: str, info: dict):
        """Fallback 1: Yahoo Finance download CSV endpoint (cloud-friendly)."""
        import pandas as pd
        import io
        yf_sym = info["yf"]
        now = int(time.time())
        period1 = now - 86400 * 3  # 3 days back
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{yf_sym}"
        params = {"period1": period1, "period2": now, "interval": "5m", "events": "history"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        resp = _requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None, None
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or "Close" not in df.columns:
            return None, None
        price = float(df["Close"].iloc[-1])
        hist = df.rename(columns=str.title)  # normalize column names
        return price, hist

    def _fetch_yahoo_chart_v8(self, sym: str, info: dict):
        """Fallback 2: Yahoo Finance chart API with cookie/crumb auth."""
        import pandas as pd
        yf_sym = info["yf"]
        sess = _requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
        # Step 1: get cookie + crumb
        try:
            r1 = sess.get("https://fc.yahoo.com", timeout=10, allow_redirects=True)
        except Exception:
            pass  # We just need the cookies
        try:
            crumb_resp = sess.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
            crumb = crumb_resp.text.strip()
        except Exception:
            crumb = ""
        # Step 2: fetch chart data
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_sym}"
        params = {"range": "2d", "interval": "5m", "crumb": crumb}
        resp = sess.get(url, params=params, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None, None
        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is None:
            return None, None
        quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
        if quotes:
            closes = [c for c in (quotes.get("close") or []) if c is not None]
            highs = [h for h in (quotes.get("high") or []) if h is not None]
            lows = [lo for lo in (quotes.get("low") or []) if lo is not None]
            vols = [v for v in (quotes.get("volume") or []) if v is not None]
            min_len = min(len(closes), len(highs), len(lows))
            if min_len >= 20:
                hist = pd.DataFrame({"Close": closes[:min_len], "High": highs[:min_len], "Low": lows[:min_len], "Volume": (vols[:min_len] if len(vols) >= min_len else [0]*min_len)})
                return float(price), hist
        return float(price), None

    def _fetch_marketaux(self, sym: str, info: dict):
        """Fallback: Yahoo Finance quote endpoint."""
        yf_sym = info["yf"]
        url = f"https://query2.finance.yahoo.com/v6/finance/quote"
        params = {"symbols": yf_sym}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        results = data.get("quoteResponse", {}).get("result", [])
        if results:
            price = results[0].get("regularMarketPrice")
            if price:
                return float(price), None
        return None, None

    # ── Cloud-Friendly Price Sources ─────────────────────────────────────────

    # Stooq symbols (BRENT not available on Stooq)
    # Note: SI.F on Stooq is quoted in cents — divide by 100
    _STOOQ_MAP = {
        "WTI": "cl.f", "XAGUSD": "si.f", "XAUUSD": "gc.f",
        "NGAS": "ng.f", "COPPER": "hg.f", "PLATINUM": "pl.f", "PALLADIUM": "pa.f",
        "EURUSD": "eurusd", "GBPUSD": "gbpusd", "USDJPY": "usdjpy",
    }
    _STOOQ_DIVISOR = {"XAGUSD": 100}  # SI.F is in cents per oz

    def _fetch_stooq(self, sym: str, info: dict):
        """Cloud-friendly: Stooq.com CSV endpoint (no auth, no IP blocking)."""
        stooq_sym = self._STOOQ_MAP.get(sym)
        if not stooq_sym or not _requests:
            return None, None
        url = f"https://stooq.com/q/l/?s={stooq_sym}&f=sd2t2ohlcv&h&e=csv"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return None, None
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            return None, None
        parts = lines[1].split(",")
        if len(parts) < 7 or parts[6] == "N/D":
            return None, None
        try:
            close_price = float(parts[6])
            if close_price <= 0:
                return None, None
            divisor = self._STOOQ_DIVISOR.get(sym, 1)
            return close_price / divisor, None
        except (ValueError, IndexError):
            return None, None

    def _fetch_yahoo_chart_simple(self, sym: str, info: dict):
        """Cloud-friendly: Yahoo chart on query1 — no crumb needed, full OHLC data."""
        import pandas as pd
        if not _requests:
            return None, None
        yf_sym = info["yf"]
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
        params = {"range": "2d", "interval": "5m"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        resp = _requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None, None
        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is None:
            return None, None
        quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
        if quotes:
            closes = [c for c in (quotes.get("close") or []) if c is not None]
            highs = [h for h in (quotes.get("high") or []) if h is not None]
            lows = [lo for lo in (quotes.get("low") or []) if lo is not None]
            vols = [v for v in (quotes.get("volume") or []) if v is not None]
            min_len = min(len(closes), len(highs), len(lows))
            if min_len >= 20:
                hist = pd.DataFrame({
                    "Close": closes[:min_len],
                    "High": highs[:min_len],
                    "Low": lows[:min_len],
                    "Volume": vols[:min_len] if len(vols) >= min_len else [0] * min_len,
                })
                return float(price), hist
        return float(price), None

    def fetch_prices(self):
        """Fetch current prices using multiple sources with fallback chain.
        Staggers requests with small delays to avoid rate limiting (Feature 13)."""
        import pandas as pd
        for idx, (sym, info) in enumerate(SYMBOLS.items()):
            # Rate limit protection: small delay between symbols to avoid Yahoo throttling
            if idx > 0:
                time.sleep(0.5)  # 500ms between symbols = ~7s total for 14 symbols
            price = None
            hist = None

            # Cloud deployment: prioritize sources that aren't blocked by Yahoo
            if self._is_cloud:
                sources = [
                    ("yahoo_chart_simple", self._fetch_yahoo_chart_simple),
                    ("stooq", self._fetch_stooq),
                    ("yahoo_chart_v8", self._fetch_yahoo_chart_v8),
                    ("yahoo_download", self._fetch_yahoo_download),
                    ("yahoo_quote", self._fetch_marketaux),
                ]
            else:
                sources = [
                    ("yfinance", self._fetch_yfinance),
                    ("yahoo_chart_simple", self._fetch_yahoo_chart_simple),
                    ("yahoo_download", self._fetch_yahoo_download),
                    ("yahoo_chart_v8", self._fetch_yahoo_chart_v8),
                    ("stooq", self._fetch_stooq),
                    ("yahoo_quote", self._fetch_marketaux),
                ]
            for source_name, fetcher in sources:
                try:
                    price, hist = self._run_with_timeout(fetcher, (sym, info), timeout_sec=20)
                    if price and price > 0:
                        self._log(f"[{sym}] price ${price:.2f} via {source_name}", level="INFO")
                        break
                except Exception as e:
                    self._log(f"{source_name} error for {sym}: {type(e).__name__}: {e}", level="WARN")
                    price, hist = None, None

            if price and price > 0:
                self.prices[sym] = price
                self._price_timestamps[sym] = time.time()  # Feature 14: track freshness
                last_high = price
                last_low = price
                if hist is not None and len(hist) > 0:
                    try:
                        last_high = float(hist["High"].iloc[-1])
                        last_low = float(hist["Low"].iloc[-1])
                    except Exception:
                        pass
                self.price_history[sym].append({
                    "price": price,
                    "time": datetime.now().isoformat(),
                    "high": last_high,
                    "low": last_low,
                    "volume": float(hist["Volume"].iloc[-1]) if (hist is not None and len(hist) > 0 and "Volume" in hist) else 0,
                })
                if hist is not None and len(hist) >= 20:
                    self._compute_indicators(sym, hist)
                elif len(self.price_history[sym]) >= 20:
                    ph = list(self.price_history[sym])
                    fallback_hist = pd.DataFrame({
                        "Close": [p["price"] for p in ph],
                        "High": [p["high"] for p in ph],
                        "Low": [p["low"] for p in ph],
                        "Volume": [0] * len(ph),
                    })
                    self._compute_indicators(sym, fallback_hist)
            else:
                self._log(f"ALL sources failed for {sym}", level="ERROR")

    def _compute_indicators(self, symbol: str, hist):
        """Compute SMA, EMA, RSI, Bollinger Bands, MACD from price history."""
        closes = hist["Close"].values
        if len(closes) < 20:
            return

        # SMA
        sma_10 = float(closes[-10:].mean())
        sma_20 = float(closes[-20:].mean())

        # EMA-12 and EMA-26
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)

        # MACD: compute full EMA series to get proper signal line
        macd_line = ema_12 - ema_26
        if len(closes) >= 35:
            # Build MACD line series for signal EMA
            macd_series = []
            for i in range(34, len(closes)):
                e12 = self._ema(closes[:i+1], 12)
                e26 = self._ema(closes[:i+1], 26)
                macd_series.append(e12 - e26)
            if len(macd_series) >= 9:
                import numpy as np
                macd_signal = self._ema(np.array(macd_series), 9)
            else:
                macd_signal = macd_line
        else:
            macd_signal = macd_line * 0.8  # approximate when not enough data
        macd_histogram = macd_line - macd_signal

        # RSI (14-period)
        rsi = self._rsi(closes, 14)

        # Bollinger Bands (20-period, 2 std)
        bb_mid = sma_20
        std_20 = float(closes[-20:].std())
        bb_upper = bb_mid + 2 * std_20
        bb_lower = bb_mid - 2 * std_20

        # ATR (14-period) for volatility-based stops
        highs = hist["High"].values
        lows = hist["Low"].values
        atr = self._atr(highs, lows, closes, 14)

        # ADX (14-period) for regime detection
        adx = self._adx(highs, lows, closes, 14)

        # Price momentum (rate of change over 5 periods)
        roc_5 = float((closes[-1] - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0.0

        current_price = float(closes[-1])

        # RSI/Price Divergence Detection (Feature 3)
        divergence = "NONE"
        if len(closes) >= 20:
            # Compare last 10 bars: price direction vs RSI direction
            price_recent = float(closes[-1])
            price_prev = float(closes[-10])
            rsi_recent = self._rsi(closes, 14)
            rsi_prev = self._rsi(closes[:-5], 14) if len(closes) > 19 else rsi_recent
            if price_recent > price_prev and rsi_recent < rsi_prev - 3:
                divergence = "BEARISH"  # price up, RSI down = bearish divergence
            elif price_recent < price_prev and rsi_recent > rsi_prev + 3:
                divergence = "BULLISH"  # price down, RSI up = bullish divergence

        # Volume analysis (Feature 4)
        volume_ratio = 1.0
        if "Volume" in hist.columns:
            vols = hist["Volume"].values
            valid_vols = [float(v) for v in vols[-20:] if v is not None and float(v) > 0]
            if len(valid_vols) >= 5:
                avg_vol = sum(valid_vols) / len(valid_vols)
                current_vol = valid_vols[-1] if valid_vols else 0
                volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # ADX trend (for breakout detection — Feature 6)
        prev_adx = self.indicators.get(symbol, {}).get("adx", adx)
        adx_rising = adx > prev_adx + 2  # ADX increasing by 2+ points

        # Track ATR history for volatility-normalized sizing
        self._atr_history[symbol].append(atr)

        self.indicators[symbol] = {
            "sma_10": sma_10,
            "sma_20": sma_20,
            "ema_12": float(ema_12),
            "ema_26": float(ema_26),
            "rsi": float(rsi),
            "macd_line": float(macd_line),
            "macd_signal": float(macd_signal),
            "macd_histogram": float(macd_histogram),
            "bb_upper": float(bb_upper),
            "bb_mid": float(bb_mid),
            "bb_lower": float(bb_lower),
            "atr": float(atr),
            "adx": float(adx),
            "adx_rising": bool(adx_rising),
            "roc_5": roc_5,
            "divergence": divergence,
            "volume_ratio": float(volume_ratio),
            "price": current_price,
            "trend": "BULLISH" if sma_10 > sma_20 else "BEARISH",
            "bb_position": float((current_price - bb_lower) / (bb_upper - bb_lower)) if bb_upper != bb_lower else 0.5,
        }

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return float(data[-1])
        multiplier = 2 / (period + 1)
        ema = float(data[-period])
        for price in data[-period + 1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    @staticmethod
    def _ema_from_values(data, period):
        if not data:
            return 0
        return data[-1]

    @staticmethod
    def _rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(-period, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0001
        avg_loss = sum(losses) / period if losses else 0.0001
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return float(highs[-1] - lows[-1]) if len(highs) > 0 else 1.0
        trs = []
        for i in range(-period, 0):
            tr = max(
                float(highs[i] - lows[i]),
                abs(float(highs[i] - closes[i - 1])),
                abs(float(lows[i] - closes[i - 1]))
            )
            trs.append(tr)
        return sum(trs) / len(trs)

    @staticmethod
    def _adx(highs, lows, closes, period=14):
        """Average Directional Index — measures trend strength (0-100).
        ADX > 25 = trending, ADX < 20 = ranging/choppy."""
        if len(highs) < period + 2:
            return 25.0  # neutral default
        n = len(highs)
        plus_dm = []
        minus_dm = []
        tr_list = []
        for i in range(1, n):
            up_move = float(highs[i] - highs[i - 1])
            down_move = float(lows[i - 1] - lows[i])
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
            tr_list.append(max(
                float(highs[i] - lows[i]),
                abs(float(highs[i] - closes[i - 1])),
                abs(float(lows[i] - closes[i - 1])),
            ))
        if len(tr_list) < period:
            return 25.0
        # Wilder smoothing
        atr_s = sum(tr_list[:period])
        plus_s = sum(plus_dm[:period])
        minus_s = sum(minus_dm[:period])
        dx_vals = []
        for i in range(period, len(tr_list)):
            atr_s = atr_s - atr_s / period + tr_list[i]
            plus_s = plus_s - plus_s / period + plus_dm[i]
            minus_s = minus_s - minus_s / period + minus_dm[i]
            plus_di = 100 * plus_s / atr_s if atr_s > 0 else 0
            minus_di = 100 * minus_s / atr_s if atr_s > 0 else 0
            di_sum = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
            dx_vals.append(dx)
        if not dx_vals:
            return 25.0
        if len(dx_vals) < period:
            return float(sum(dx_vals) / len(dx_vals))
        adx = sum(dx_vals[:period]) / period
        for i in range(period, len(dx_vals)):
            adx = (adx * (period - 1) + dx_vals[i]) / period
        return float(adx)

    # ── Adaptive Parameters ────────────────────────────────────────────────
    def _get_symbol_param(self, symbol: str, param: str, default: float) -> float:
        """Get an adaptive parameter for a symbol, falling back to global default."""
        sp = self.symbol_params.get(symbol)
        if sp:
            return sp.get(param, default)
        return default

    def _run_background_optimization(self):
        """Run walk-forward optimization for top symbols and update adaptive params.
        On cloud free tier, uses reduced grid and fewer symbols to avoid OOM.
        """
        import backtest as bt
        import gc
        is_cloud = bool(os.environ.get("RENDER_EXTERNAL_URL"))

        # On free tier, only optimize the top 3 most-traded symbols with reduced grid
        if is_cloud:
            trade_counts = {}
            for t in self.trade_history:
                sym = t.get("symbol", "")
                trade_counts[sym] = trade_counts.get(sym, 0) + 1
            top_syms = sorted(trade_counts, key=trade_counts.get, reverse=True)[:3]
            if not top_syms:
                top_syms = ["XAUUSD", "BRENT", "EURUSD"]
            saved_grid = bt.PARAM_GRID.copy()
            bt.PARAM_GRID = {
                "signal_threshold": [1.5, 2.5],
                "atr_stop_mult":    [2.0],
                "risk_reward":      [1.5, 2.5],
                "max_risk_pct":     [0.20],
            }
            opt_days = 90
        else:
            top_syms = list(SYMBOLS.keys())
            opt_days = 180

        self._log(f"Starting background optimization for {len(top_syms)} symbols...", level="INFO")
        for sym in top_syms:
            try:
                gc.collect()
                result = bt.run_optimization(sym, period_days=opt_days)
                if "best_params" in result and result["best_params"]:
                    self.symbol_params[sym] = result["best_params"]
                    self._log(f"[ADAPTIVE] {sym} optimized: {result['best_params']}", level="INFO")
            except Exception as e:
                self._log(f"[ADAPTIVE] Optimization failed for {sym}: {e}", level="WARN")

        if is_cloud:
            bt.PARAM_GRID = saved_grid

        self._last_optimization_run = time.time()
        self._log("Background optimization complete.", level="INFO")

    # ── Session Detection ──────────────────────────────────────────────────
    @staticmethod
    def _get_current_session() -> dict:
        """Determine which trading session is active based on UTC time."""
        from datetime import timezone
        utc_hour = datetime.now(timezone.utc).hour
        for name, cfg in SESSIONS.items():
            start, end = cfg["hours"]
            if start <= utc_hour < end:
                return {
                    "name": name,
                    "label": cfg["label"],
                    "boost": cfg["boost"],
                    "class_mult": cfg["class_mult"],
                    "utc_hour": utc_hour,
                }
        # Fallback (shouldn't happen since sessions cover 0-24)
        return {"name": "OFF_HOURS", "label": "Off-Hours", "boost": 0.0,
                "class_mult": {"oil": 0.0, "energy": 0.0, "metals": 0.0, "forex": 0.3, "indices": 0.0},
                "utc_hour": utc_hour}

    def _get_session_multiplier(self, symbol: str) -> float:
        """Get position size multiplier for a symbol based on current session."""
        session = self._get_current_session()
        asset_class = SYMBOLS.get(symbol, {}).get("asset_class", "metals")
        return session.get("class_mult", {}).get(asset_class, 1.0)

    # ── Risk Management ─────────────────────────────────────────────────────
    def _calculate_position_size(self, symbol: str, stop_distance: float) -> float:
        """Calculate position size based on risk budget and stop distance."""
        risk_budget = self.equity * (MAX_RISK_PER_TRADE_PCT / 100)
        # Reduce risk if close to floor
        buffer = self.equity - self.hard_floor
        if buffer < 500:
            risk_budget *= 0.3   # drastically reduce near floor
        elif buffer < 750:
            risk_budget *= 0.5
        elif buffer < 1000:
            risk_budget *= 0.7

        # Equity curve trading: cut size by 50% when recent trades are net negative
        if len(self._recent_trades_pnl) >= 5:
            rolling_avg = sum(self._recent_trades_pnl) / len(self._recent_trades_pnl)
            if rolling_avg < 0:
                risk_budget *= 0.5

        # Volatility-Normalized Sizing (Feature 7): scale down when ATR is elevated
        atr_hist = self._atr_history.get(symbol)
        if atr_hist and len(atr_hist) >= 10:
            median_atr = sorted(atr_hist)[len(atr_hist) // 2]
            current_atr = atr_hist[-1]
            if median_atr > 0 and current_atr > median_atr * 1.5:
                # ATR is 50%+ above median — reduce size proportionally
                vol_scale = median_atr / current_atr  # e.g. 0.67 if ATR is 1.5x median
                vol_scale = max(vol_scale, 0.3)  # never less than 30%
                risk_budget *= vol_scale

        if stop_distance <= 0:
            return 0

        lot_size = SYMBOLS[symbol]["lot_size"]
        raw_qty = risk_budget / stop_distance
        # Round to lot_size
        qty = max(lot_size, round(raw_qty / lot_size) * lot_size)
        # Cap max position value at 20% of equity
        max_value = self.equity * 0.20
        price = self.prices.get(symbol, 0)
        if price > 0 and qty * price > max_value:
            qty = max(lot_size, int(max_value / price / lot_size) * lot_size)
        return qty

    def _can_open_position(self, symbol: str, side: str = None) -> tuple[bool, str]:
        """Check if we're allowed to open a new position."""
        # Weekend guard: markets closed Saturday & Sunday (UTC)
        # Forex opens Sunday 22:00 UTC, closes Friday 22:00 UTC
        # Commodities/Indices: closed all of Saturday & Sunday
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        day_of_week = now.weekday()  # 0=Mon, 5=Sat, 6=Sun
        utc_hour = now.hour
        asset_class = SYMBOLS.get(symbol, {}).get("asset_class", "")
        if asset_class == "forex":
            # Forex: closed from Friday 22:00 UTC to Sunday 22:00 UTC
            if day_of_week == 5:  # Saturday — always closed
                return False, "Forex closed (weekend)"
            if day_of_week == 4 and utc_hour >= 22:  # Friday after 22:00
                return False, "Forex closed (weekend — Friday close)"
            if day_of_week == 6 and utc_hour < 22:  # Sunday before 22:00
                return False, "Forex closed (weekend — opens Sunday 22:00 UTC)"
        else:
            if day_of_week >= 5:
                return False, "Markets closed (weekend)"

        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False, "Max open positions reached"

        sym_count = sum(1 for p in self.positions.values() if p.symbol == symbol)
        if sym_count >= MAX_POSITIONS_PER_SYMBOL:
            return False, f"Max positions for {symbol} reached"

        # Asset class concentration limit
        asset_class = SYMBOLS.get(symbol, {}).get("asset_class", "")
        class_count = sum(1 for p in self.positions.values()
                         if SYMBOLS.get(p.symbol, {}).get("asset_class") == asset_class)
        if class_count >= MAX_POSITIONS_PER_CLASS:
            return False, f"Max positions for {asset_class} class reached ({class_count}/{MAX_POSITIONS_PER_CLASS})"

        if self.daily_pnl <= -MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"

        # Pre-check: ensure opening won't risk breaching floor
        buffer = self.equity - self.hard_floor
        if buffer < 200:
            return False, "Too close to hard floor, pausing new trades"

        # Trade cooldown: don't re-enter a symbol too soon after ANY trade
        trade_cd = self._trade_cooldowns.get(symbol, 0)
        if time.time() < trade_cd:
            remaining = int(trade_cd - time.time())
            return False, f"Trade cooldown: {remaining}s remaining for {symbol}"

        # Loss cooldown: extra cooldown after a losing trade
        cooldown_until = self._loss_cooldowns.get(symbol, 0)
        if time.time() < cooldown_until:
            remaining = int(cooldown_until - time.time())
            return False, f"Loss cooldown: {remaining}s remaining for {symbol}"

        # Price Staleness Guard (Feature 14): reject if price data is too old
        last_update = self._price_timestamps.get(symbol, 0)
        if last_update > 0 and (time.time() - last_update) > 120:  # 2 minutes stale
            return False, f"Stale price data for {symbol} ({int(time.time() - last_update)}s old)"

        # Consecutive Loss Breaker (Feature 9): pause trading after 3+ consecutive losses
        if time.time() < self._loss_breaker_until:
            remaining = int(self._loss_breaker_until - time.time())
            return False, f"Loss breaker active: {remaining}s remaining ({self._consecutive_losses} consecutive losses)"

        # Max Portfolio Heat (Feature 8): total open risk capped at 1% of equity
        total_risk = 0.0
        for pos in self.positions.values():
            if pos.side == Side.BUY:
                risk = (pos.entry_price - pos.stop_loss) * pos.quantity
            else:
                risk = (pos.stop_loss - pos.entry_price) * pos.quantity
            risk = self._convert_pnl_to_usd(pos.symbol, risk, self.prices.get(pos.symbol, pos.entry_price))
            total_risk += max(0, risk)
        max_heat = self.equity * 0.01  # 1% of equity
        if total_risk >= max_heat:
            return False, f"Portfolio heat limit: ${total_risk:.0f} risk >= 1% cap (${max_heat:.0f})"

        # Correlation guard: block same-direction trades on correlated pairs
        if side:
            for pair in CORRELATED_PAIRS:
                if symbol in pair:
                    partner = (pair - {symbol}).pop()
                    for pos in self.positions.values():
                        if pos.symbol == partner and pos.side.value == side:
                            return False, f"Correlation block: already {side} on {partner}"

        return True, "OK"

    def _check_floor_emergency(self):
        """Emergency close all positions if equity near hard floor."""
        if self.equity <= self.hard_floor + 100:
            self._log("EMERGENCY: Equity near hard floor! Closing all positions.", level="CRITICAL")
            for pid in list(self.positions.keys()):
                self._close_position(pid, "EMERGENCY: Hard floor protection")

    def _update_ratcheting_floor(self):
        """Dynamic floor: as balance grows past milestones, ratchet the floor up to lock in gains."""
        # For every $500 gained above starting balance, raise the floor by $400
        # e.g. balance hits $25,500 → floor moves from $24,000 to $24,400
        # e.g. balance hits $26,000 → floor moves to $24,800
        gains_above_start = self.balance - self.starting_balance
        if gains_above_start <= 0:
            return
        milestones_hit = int(gains_above_start / 500)
        new_floor = HARD_FLOOR + (milestones_hit * 400)
        if new_floor > self.hard_floor:
            old_floor = self.hard_floor
            self.hard_floor = new_floor
            self._log(
                f"FLOOR RATCHET: ${old_floor:.0f} -> ${new_floor:.0f} (balance=${self.balance:.2f}, locked ${new_floor - HARD_FLOOR:.0f} in gains)",
                level="TRADE",
            )

    # ── Trading Strategy ────────────────────────────────────────────────────
    def evaluate_strategy(self):
        """Main strategy loop: evaluate signals and manage positions."""
        self._update_equity()
        self._check_floor_emergency()
        self._check_stops_and_targets()
        self._check_time_exits()
        self._trail_stops()

        summaries = []
        session = self._get_current_session()
        for symbol in SYMBOLS:
            if symbol not in self.indicators or symbol not in self.prices:
                continue

            ind = self.indicators[symbol]
            price = self.prices[symbol]

            signal = self._generate_signal(symbol, ind, price)
            sig_str = f"{signal['side'].value}(str={signal['strength']:.1f})" if signal else "NONE"
            summaries.append(f"{symbol}={sig_str}")

            # Trade Journal: log every signal evaluation
            # Note: convert all numeric values to native Python types for PostgreSQL compatibility
            journal_entry = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "price": float(price) if price is not None else None,
                "adx": float(ind.get("adx", 0)),
                "rsi": float(ind.get("rsi", 50)),
                "session_name": session["name"],
            }
            if not signal:
                continue

            journal_entry["side"] = signal["side"].value
            journal_entry["strength"] = float(signal["strength"])
            journal_entry["reasons"] = "; ".join(signal["reasons"])

            can_open, reason = self._can_open_position(symbol, side=signal["side"].value)
            if not can_open:
                summaries[-1] += f"[blocked:{reason}]"
                journal_entry["action"] = f"BLOCKED:{reason}"
                try:
                    db.save_signal(journal_entry)
                except Exception as e:
                    self._log(f"Journal save error (blocked): {e}", level="ERROR")
                continue

            journal_entry["action"] = "EXECUTED"
            try:
                db.save_signal(journal_entry)
            except Exception as e:
                self._log(f"Journal save error (executed): {e}", level="ERROR")

            # AI Signal Enhancement
            ai_result = None
            if signal and self.ai_brain.enabled:
                composite = signal["strength"] if signal["side"] == Side.BUY else -signal["strength"]
                if self.ai_brain.should_call_ai_signal(symbol, composite, ind):
                    price_hist = list(self.price_history[symbol])[-20:]
                    ai_result = self.ai_brain.enhance_signal(
                        symbol, SYMBOLS[symbol]["name"], composite,
                        signal["side"].value, ind, price_hist, signal["reasons"],
                    )
                    if ai_result and ai_result["ai_used"]:
                        if ai_result["action"] == "HOLD" and ai_result["confidence"] > 70:
                            self._log(f"[AI] Blocked {signal['side'].value} on {symbol}: {ai_result['reasoning']}", level="TRADE")
                            summaries[-1] += f"[AI:HOLD@{ai_result['confidence']}%]"
                            signal = None
                        elif ai_result["action"] == signal["side"].value:
                            signal["reasons"].append(f"AI confirms ({ai_result['confidence']}%): {ai_result['reasoning']}")
                            signal["ai_confidence"] = ai_result["confidence"]
                            summaries[-1] += f"[AI:OK@{ai_result['confidence']}%]"
                        elif ai_result["confidence"] > 70:
                            new_side = Side.BUY if ai_result["action"] == "BUY" else Side.SELL
                            signal["side"] = new_side
                            signal["reasons"].append(f"AI override ({ai_result['confidence']}%): {ai_result['reasoning']}")
                            signal["ai_confidence"] = ai_result["confidence"]
                            summaries[-1] += f"[AI:{ai_result['action']}@{ai_result['confidence']}%]"

            if signal:
                self._execute_signal(symbol, signal, ind, price, ai_result=ai_result)

        self._log(f"Strategy scan [{session['name']}]: {' | '.join(summaries)} | Equity=${self.equity:.2f} | Buffer=${self.equity - self.hard_floor:.2f}")

    def _generate_signal(self, symbol: str, ind: dict, price: float) -> dict | None:
        """
        Multi-factor signal generation:
        1. Trend following: SMA crossover + MACD confirmation
        2. Mean reversion: Bollinger Band bounce + RSI extremes
        3. Momentum: ROC confirmation
        4. Strong individual signals (extreme RSI / BB)
        """
        reasons = []

        # ─── Factor 1: Trend (SMA crossover + MACD) ────────────────────
        trend_score = 0
        if ind["sma_10"] > ind["sma_20"]:
            trend_score = 0.5
            reasons.append(f"SMA10({ind['sma_10']:.2f})>SMA20({ind['sma_20']:.2f})")
            if ind["macd_histogram"] > 0:
                trend_score = 1
                reasons[-1] += ", MACD histogram bullish"
        elif ind["sma_10"] < ind["sma_20"]:
            trend_score = -0.5
            reasons.append(f"SMA10({ind['sma_10']:.2f})<SMA20({ind['sma_20']:.2f})")
            if ind["macd_histogram"] < 0:
                trend_score = -1
                reasons[-1] += ", MACD histogram bearish"

        # ─── Factor 2: Mean Reversion (Bollinger + RSI) ─────────────────
        mr_score = 0
        if ind["rsi"] < 25 and ind["bb_position"] < 0.15:
            mr_score = 1.5   # very strong oversold
            reasons.append(f"Strongly oversold: RSI={ind['rsi']:.1f}, BB={ind['bb_position']:.2f}")
        elif ind["bb_position"] < 0.25 and ind["rsi"] < 40:
            mr_score = 1
            reasons.append(f"Oversold: BB pos={ind['bb_position']:.2f}, RSI={ind['rsi']:.1f}")
        elif ind["rsi"] > 75 and ind["bb_position"] > 0.85:
            mr_score = -1.5  # very strong overbought
            reasons.append(f"Strongly overbought: RSI={ind['rsi']:.1f}, BB={ind['bb_position']:.2f}")
        elif ind["bb_position"] > 0.75 and ind["rsi"] > 60:
            mr_score = -1
            reasons.append(f"Overbought: BB pos={ind['bb_position']:.2f}, RSI={ind['rsi']:.1f}")

        # ─── Factor 3: Momentum (ROC) ──────────────────────────────────
        mom_score = 0
        if ind["roc_5"] > 0.2:
            mom_score = 1
            reasons.append(f"Positive momentum: ROC5={ind['roc_5']:.2f}%")
        elif ind["roc_5"] < -0.2:
            mom_score = -1
            reasons.append(f"Negative momentum: ROC5={ind['roc_5']:.2f}%")

        # ─── Composite Signal ──────────────────────────────────────────
        total = trend_score + mr_score + mom_score

        # ─── Factor 4: MACD crossover strength ─────────────────────────
        macd_score = 0
        if ind["macd_histogram"] > 0 and ind["macd_line"] > 0:
            macd_score = 0.5
            reasons.append(f"MACD bullish (line={ind['macd_line']:.3f}, hist={ind['macd_histogram']:.3f})")
        elif ind["macd_histogram"] < 0 and ind["macd_line"] < 0:
            macd_score = -0.5
            reasons.append(f"MACD bearish (line={ind['macd_line']:.3f}, hist={ind['macd_histogram']:.3f})")

        # ─── Factor 5: News Sentiment (AI-powered) ─────────────────────
        sentiment_score = 0
        sent_data = self.news_sentiment.get(symbol, {})
        sent_val = sent_data.get("score", 0.0)
        if abs(sent_val) > 0.3:
            sentiment_score = sent_val * 0.75
            reasons.append(f"News sentiment: {sent_val:+.2f}")

        # ─── Factor 6: Higher Timeframe Filter (1H) ─────────────────
        htf_score = 0
        htf = self.htf_indicators.get(symbol)
        if htf:
            htf_trend = htf.get("trend", "NEUTRAL")
            htf_rsi = htf.get("rsi", 50)
            if htf_trend == "BULLISH" and htf_rsi < 70:
                htf_score = 1.0
                reasons.append(f"HTF(1h): BULLISH, RSI={htf_rsi:.1f}")
            elif htf_trend == "LEAN_BULLISH":
                htf_score = 0.5
                reasons.append(f"HTF(1h): Lean bullish, RSI={htf_rsi:.1f}")
            elif htf_trend == "BEARISH" and htf_rsi > 30:
                htf_score = -1.0
                reasons.append(f"HTF(1h): BEARISH, RSI={htf_rsi:.1f}")
            elif htf_trend == "LEAN_BEARISH":
                htf_score = -0.5
                reasons.append(f"HTF(1h): Lean bearish, RSI={htf_rsi:.1f}")

        # ─── Factor 7: Session Boost ────────────────────────────────
        session = self._get_current_session()
        session_boost = session["boost"]
        if session_boost > 0:
            # Boost in the direction of the dominant signal
            if trend_score + mr_score + mom_score > 0:
                reasons.append(f"Session boost: {session['name']} (+{session_boost})")
            elif trend_score + mr_score + mom_score < 0:
                session_boost = -session_boost
                reasons.append(f"Session boost: {session['name']} ({session_boost})")
            else:
                session_boost = 0  # No boost if no directional bias

        # ─── Factor 8: RSI/Price Divergence (Feature 3) ──────────────
        div_score = 0
        divergence = ind.get("divergence", "NONE")
        if divergence == "BULLISH":
            div_score = 1.0
            reasons.append(f"Bullish RSI divergence (price down, RSI up)")
        elif divergence == "BEARISH":
            div_score = -1.0
            reasons.append(f"Bearish RSI divergence (price up, RSI down)")

        # ─── Factor 9: Volume Confirmation (Feature 4) ───────────────
        volume_ratio = ind.get("volume_ratio", 1.0)
        volume_filter = 1.0
        if volume_ratio > 1.5:
            volume_filter = 1.2  # high volume confirms move
            reasons.append(f"High volume confirmation ({volume_ratio:.1f}x avg)")
        elif volume_ratio < 0.5:
            volume_filter = 0.6  # low volume weakens signal
            reasons.append(f"Low volume warning ({volume_ratio:.1f}x avg)")

        # ─── Factor 10: Breakout Detection (Feature 6) ───────────────
        breakout_score = 0
        adx_rising = ind.get("adx_rising", False)
        if adx_rising and ind.get("adx", 25) > 20:
            if trend_score > 0:
                breakout_score = 0.5
                reasons.append(f"Bullish breakout (ADX rising + trend)")
            elif trend_score < 0:
                breakout_score = -0.5
                reasons.append(f"Bearish breakout (ADX rising + trend)")

        total = (trend_score + mr_score + mom_score + macd_score + sentiment_score
                 + htf_score + session_boost + div_score + breakout_score) * volume_filter

        # ─── ADX Regime Filter (applied before signal decision) ────
        adx = ind.get("adx", 25)
        if adx < 18:
            # Very choppy market — don't trade at all (was 20, only blocking weak MR)
            return None
        elif adx < 22:
            reasons.append(f"ADX={adx:.1f} (ranging) — mean-reversion only")
            # In ranging markets, only allow if strong mean-reversion
            if abs(mr_score) < 1.5:
                return None
        elif adx > 25:
            reasons.append(f"ADX={adx:.1f} (trending) — trend-following mode")
        else:
            reasons.append(f"ADX={adx:.1f} (transitional)")

        # ─── HTF MANDATORY alignment (not just a filter) ──────────
        # This is the #1 change: REQUIRE the 1H trend to agree with our direction
        htf = self.htf_indicators.get(symbol)
        if not htf:
            return None  # no HTF data = no trade
        htf_trend = htf.get("trend", "NEUTRAL")
        htf_rsi = htf.get("rsi", 50)

        # Determine if HTF supports a BUY or SELL
        htf_bullish = htf_trend in ("BULLISH", "LEAN_BULLISH")
        htf_bearish = htf_trend in ("BEARISH", "LEAN_BEARISH")

        # Adaptive signal threshold per symbol
        threshold = self._get_symbol_param(symbol, "signal_threshold", SIGNAL_THRESHOLD)

        # Standard composite: need strong agreement from multiple factors
        if total >= threshold:
            if not htf_bullish:
                return None  # signal says BUY but HTF doesn't agree
            signal = {"side": Side.BUY, "strength": total, "reasons": reasons}
        elif total <= -threshold:
            if not htf_bearish:
                return None  # signal says SELL but HTF doesn't agree
            signal = {"side": Side.SELL, "strength": abs(total), "reasons": reasons}
        else:
            return None
        # NOTE: removed the weak "shortcut" entries (MR-alone and Trend+MACD-alone)
        # Those were letting in low-conviction trades that bled money

        reasons.append(f"HTF(1h) confirms: {htf_trend}, RSI={htf_rsi:.1f}")
        return signal

    def _execute_signal(self, symbol: str, signal: dict, ind: dict, price: float, ai_result: dict = None):
        """Open a position based on the signal with ATR-based stops."""
        atr = ind["atr"]
        side = signal["side"]

        # Adaptive per-symbol parameters (from walk-forward optimization)
        atr_stop = self._get_symbol_param(symbol, "atr_stop_mult", ATR_STOP_MULT)
        rr_ratio = self._get_symbol_param(symbol, "risk_reward", RISK_REWARD_RATIO)

        # ATR-based stop loss and take profit
        stop_distance = atr * atr_stop
        tp_distance = stop_distance * rr_ratio

        if side == Side.BUY:
            stop_loss = price - stop_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + stop_distance
            take_profit = price - tp_distance

        quantity = self._calculate_position_size(symbol, stop_distance)
        if quantity <= 0:
            return

        # Session-based position sizing
        session_mult = self._get_session_multiplier(symbol)
        if session_mult <= 0:
            self._log(f"Skipping {symbol} trade: session {self._get_current_session()['name']} blocks this asset", level="INFO")
            return
        if session_mult != 1.0:
            lot_size = SYMBOLS[symbol]["lot_size"]
            quantity = max(lot_size, round(quantity * session_mult / lot_size) * lot_size)

        # Final safety: simulate worst case
        worst_loss = quantity * stop_distance
        if self.equity - worst_loss < self.hard_floor:
            self._log(f"Skipping {symbol} trade: worst-case loss ${worst_loss:.2f} would breach floor", level="WARN")
            return

        pos_id = str(uuid.uuid4())[:8]
        reason = f"{side.value} signal (score={signal['strength']}): " + "; ".join(signal["reasons"])

        position = Position(
            id=pos_id,
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            reason=reason,
            opened_at=datetime.now().isoformat(),
            original_quantity=quantity,
        )

        with self._lock:
            self.positions[pos_id] = position
            self.total_trades += 1

        risk_amt = quantity * stop_distance
        reward_amt = quantity * tp_distance

        self._log(
            f"OPENED {side.value} {symbol} @ {price:.4f} | "
            f"Qty: {quantity} | SL: {position.stop_loss} | TP: {position.take_profit} | "
            f"Risk: ${risk_amt:.2f} | Reward: ${reward_amt:.2f} | "
            f"Reason: {reason}",
            level="TRADE",
            trade_data={
                "action": "OPEN",
                "id": pos_id,
                "symbol": symbol,
                "side": side.value,
                "price": price,
                "quantity": quantity,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "risk": round(risk_amt, 2),
                "reward": round(reward_amt, 2),
                "reason": reason,
                "ai_confidence": signal.get("ai_confidence", 0),
            }
        )

        # AI explanation in background thread
        if self.ai_brain.enabled:
            def _ai_explain():
                explanation = self.ai_brain.explain_trade_open(
                    symbol, SYMBOLS[symbol]["name"], side.value, price,
                    ind, signal["reasons"], ai_result,
                )
                if explanation:
                    position.reason = explanation
                    self.ai_explanations[pos_id] = explanation
            threading.Thread(target=_ai_explain, daemon=True).start()

    def _check_stops_and_targets(self):
        """Check all open positions for SL/TP hits and partial profit taking."""
        for pid in list(self.positions.keys()):
            pos = self.positions.get(pid)
            if not pos:
                continue

            price = self.prices.get(pos.symbol)
            if not price:
                continue

            # Partial profit taking: close 50% at 1.5x ATR profit
            if not pos.partial_closed:
                atr = self.indicators.get(pos.symbol, {}).get("atr", 0)
                if atr > 0:
                    partial_target = atr * 1.5
                    if pos.side == Side.BUY and price - pos.entry_price >= partial_target:
                        self._take_partial_profit(pos, price, partial_target)
                    elif pos.side == Side.SELL and pos.entry_price - price >= partial_target:
                        self._take_partial_profit(pos, price, partial_target)

            if pos.side == Side.BUY:
                if price <= pos.stop_loss:
                    self._close_position(pid, f"Stop Loss hit @ {price:.4f} (SL was {pos.stop_loss})")
                elif price >= pos.take_profit:
                    self._close_position(pid, f"Take Profit hit @ {price:.4f} (TP was {pos.take_profit})")
            else:  # SELL
                if price >= pos.stop_loss:
                    self._close_position(pid, f"Stop Loss hit @ {price:.4f} (SL was {pos.stop_loss})")
                elif price <= pos.take_profit:
                    self._close_position(pid, f"Take Profit hit @ {price:.4f} (TP was {pos.take_profit})")

    def _take_partial_profit(self, pos: Position, price: float, partial_target: float):
        """Close 50% of a position at 1.5x ATR profit and move SL to break-even."""
        lot_size = SYMBOLS[pos.symbol]["lot_size"]
        close_qty = max(lot_size, round((pos.quantity * 0.5) / lot_size) * lot_size)
        if close_qty >= pos.quantity:
            return  # can't close more than we have

        if pos.side == Side.BUY:
            pnl = (price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - price) * close_qty
        pnl = self._convert_pnl_to_usd(pos.symbol, pnl, price)

        with self._lock:
            pos.quantity -= close_qty
            pos.partial_closed = True
            pos.stop_loss = pos.entry_price  # move SL to break-even
            self.balance += pnl
            self.daily_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
            self.total_trades += 1

        self._update_ratcheting_floor()

        self._log(
            f"PARTIAL CLOSE {pos.side.value} {pos.symbol} @ {price:.4f} | "
            f"Closed {close_qty} of {pos.original_quantity} | PnL: ${pnl:+.2f} | "
            f"SL moved to break-even ({pos.entry_price:.4f})",
            level="TRADE",
            trade_data={
                "action": "PARTIAL_CLOSE",
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side.value,
                "price": price,
                "quantity_closed": close_qty,
                "quantity_remaining": pos.quantity,
                "pnl": round(pnl, 2),
            }
        )
        self._save_state_to_db()

    def _trail_stops(self):
        """Trail stop losses — only activates after price moves ATR_TRAIL_TRIGGER in our favour.
        This prevents locking in break-even on small moves while real wins reach TP naturally."""
        for pos in list(self.positions.values()):
            price = self.prices.get(pos.symbol)
            if not price:
                continue
            atr = self.indicators.get(pos.symbol, {}).get("atr", 0)
            if atr <= 0:
                continue

            trigger_dist = atr * ATR_TRAIL_TRIGGER  # price must move this far before trailing starts
            trail_dist = atr * ATR_TRAIL_DIST        # trail follows this far behind price

            if pos.side == Side.BUY:
                profit_dist = price - pos.entry_price
                if profit_dist < trigger_dist:
                    continue  # not enough profit yet — let TP do its job
                new_sl = price - trail_dist
                if new_sl > pos.stop_loss and new_sl > pos.entry_price:
                    old_sl = pos.stop_loss
                    pos.stop_loss = round(new_sl, 4)
                    self._log(f"Trailing SL {pos.symbol} BUY: {old_sl:.4f} -> {pos.stop_loss} (profit locked: +{profit_dist:.4f})")
            else:
                profit_dist = pos.entry_price - price
                if profit_dist < trigger_dist:
                    continue  # not enough profit yet
                new_sl = price + trail_dist
                if new_sl < pos.stop_loss and new_sl < pos.entry_price:
                    old_sl = pos.stop_loss
                    pos.stop_loss = round(new_sl, 4)
                    self._log(f"Trailing SL {pos.symbol} SELL: {old_sl:.4f} -> {pos.stop_loss} (profit locked: +{profit_dist:.4f})")

    def _check_time_exits(self):
        """Close trades that haven't moved 1x ATR in our favor within 7 hours."""
        now = datetime.now()
        for pid in list(self.positions.keys()):
            pos = self.positions.get(pid)
            if not pos:
                continue
            try:
                opened = datetime.fromisoformat(pos.opened_at)
                elapsed = (now - opened).total_seconds()
            except Exception:
                continue
            if elapsed < 7 * 3600:  # less than 7 hours
                continue
            atr = self.indicators.get(pos.symbol, {}).get("atr", 0)
            if atr <= 0:
                continue
            price = self.prices.get(pos.symbol)
            if not price:
                continue
            if pos.side == Side.BUY:
                move = price - pos.entry_price
            else:
                move = pos.entry_price - price
            if move < atr:  # hasn't moved 1x ATR in our favor
                hours = elapsed / 3600
                self._close_position(pid, f"Time exit: {hours:.1f}h elapsed, only {move:.4f} move vs {atr:.4f} ATR target")

    def _convert_pnl_to_usd(self, symbol: str, pnl: float, exit_price: float) -> float:
        """Convert PnL to USD for non-USD denominated pairs (e.g., USDJPY)."""
        info = SYMBOLS.get(symbol, {})
        if info.get("pnl_ccy") == "JPY" and exit_price > 0:
            return pnl / exit_price  # convert JPY PnL to USD
        return pnl

    def _close_position(self, pos_id: str, reason: str):
        """Close a position and record the trade."""
        with self._lock:
            pos = self.positions.pop(pos_id, None)
        if not pos:
            return

        exit_price = self.prices.get(pos.symbol, pos.entry_price)

        if pos.side == Side.BUY:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
        pnl = self._convert_pnl_to_usd(pos.symbol, pnl, exit_price)

        with self._lock:
            self.balance += pnl
            self.daily_pnl += pnl
            self._recent_trades_pnl.append(pnl)  # equity curve tracking
            # Universal trade cooldown: 30 min per symbol after ANY trade
            self._trade_cooldowns[pos.symbol] = time.time() + TRADE_COOLDOWN_SEC
            if pnl > 0:
                self.winning_trades += 1
                self._consecutive_losses = 0  # reset streak on win
            else:
                # Loss cooldown: prevent re-entry on this symbol for a while
                self._loss_cooldowns[pos.symbol] = time.time() + LOSS_COOLDOWN_SEC
                # Consecutive Loss Breaker (Feature 9)
                self._consecutive_losses += 1
                if self._consecutive_losses >= 3:
                    cooldown = 600 * (self._consecutive_losses - 2)  # 10min, 20min, 30min...
                    cooldown = min(cooldown, 3600)  # cap at 1 hour
                    self._loss_breaker_until = time.time() + cooldown
                    self._log(
                        f"LOSS BREAKER: {self._consecutive_losses} consecutive losses — pausing {cooldown // 60}min",
                        level="WARN"
                    )

        # Dynamic ratcheting floor: lock in gains as balance grows
        self._update_ratcheting_floor()

        closed = ClosedTrade(
            id=pos_id,
            symbol=pos.symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=round(pnl, 2),
            reason_open=pos.reason,
            reason_close=reason,
            opened_at=pos.opened_at,
            closed_at=datetime.now().isoformat(),
        )
        self.closed_trades.append(closed)

        # Persist to database
        try:
            db.save_trade(closed)
            self._save_state_to_db()
        except Exception as e:
            self._log(f"DB save error: {e}", level="WARN")

        self._log(
            f"CLOSED {pos.side.value} {pos.symbol} @ {exit_price:.4f} | "
            f"Entry: {pos.entry_price:.4f} | PnL: ${pnl:+.2f} | Reason: {reason}",
            level="TRADE",
            trade_data={
                "action": "CLOSE",
                "id": pos_id,
                "symbol": pos.symbol,
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "reason": reason,
            }
        )

        # AI explanation for close in background
        if self.ai_brain.enabled:
            def _ai_close_explain():
                duration = ""
                try:
                    opened = datetime.fromisoformat(pos.opened_at)
                    dur = datetime.now() - opened
                    duration = f"{dur.seconds // 3600}h {(dur.seconds % 3600) // 60}m"
                except Exception:
                    pass
                explanation = self.ai_brain.explain_trade_close(
                    pos.symbol, SYMBOLS[pos.symbol]["name"], pos.side.value,
                    pos.entry_price, exit_price, pnl, reason, duration,
                )
                if explanation:
                    closed.reason_close = explanation
            threading.Thread(target=_ai_close_explain, daemon=True).start()

        # Clean up AI explanation for this position
        self.ai_explanations.pop(pos_id, None)

    def _update_equity(self):
        """Recalculate equity = balance + unrealized PnL."""
        unrealized = 0
        for pos in self.positions.values():
            price = self.prices.get(pos.symbol, pos.entry_price)
            if pos.side == Side.BUY:
                raw_pnl = (price - pos.entry_price) * pos.quantity
            else:
                raw_pnl = (pos.entry_price - price) * pos.quantity
            pos.unrealized_pnl = self._convert_pnl_to_usd(pos.symbol, raw_pnl, price)
            unrealized += pos.unrealized_pnl

        self.equity = self.balance + unrealized
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        # Save state to DB every 10th update (~30 seconds)
        self._save_counter += 1
        if self._save_counter >= 10:
            self._save_counter = 0
            self._save_state_to_db()

    # ── Logging ─────────────────────────────────────────────────────────────
    def _log(self, message: str, level: str = "INFO", trade_data: dict = None):
        entry = {
            "time": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        if trade_data:
            entry["trade"] = trade_data
        self.trade_log.append(entry)
        # Keep log bounded
        if len(self.trade_log) > 500:
            self.trade_log = self.trade_log[-300:]
        print(f"[{level}] {message}")

    # ── Analytics (Features 10, 11, 12) ─────────────────────────────────────
    def _get_performance_attribution(self) -> dict:
        """Feature 10: P&L breakdown by symbol, asset class, session, and day of week."""
        by_symbol = {}
        by_class = {}
        by_day = {}  # 0=Mon..6=Sun
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for t in self.closed_trades:
            # By symbol
            by_symbol.setdefault(t.symbol, {"pnl": 0, "count": 0, "wins": 0})
            by_symbol[t.symbol]["pnl"] += t.pnl
            by_symbol[t.symbol]["count"] += 1
            if t.pnl > 0:
                by_symbol[t.symbol]["wins"] += 1

            # By asset class
            ac = SYMBOLS.get(t.symbol, {}).get("asset_class", "unknown")
            by_class.setdefault(ac, {"pnl": 0, "count": 0, "wins": 0})
            by_class[ac]["pnl"] += t.pnl
            by_class[ac]["count"] += 1
            if t.pnl > 0:
                by_class[ac]["wins"] += 1

            # By day of week
            try:
                dt = datetime.fromisoformat(t.closed_at)
                dow = dt.weekday()
                key = day_names[dow]
                by_day.setdefault(key, {"pnl": 0, "count": 0, "wins": 0})
                by_day[key]["pnl"] += t.pnl
                by_day[key]["count"] += 1
                if t.pnl > 0:
                    by_day[key]["wins"] += 1
            except Exception:
                pass

        # Round and add win rates
        for d in [by_symbol, by_class, by_day]:
            for v in d.values():
                v["pnl"] = round(v["pnl"], 2)
                v["win_rate"] = round(v["wins"] / v["count"] * 100, 1) if v["count"] > 0 else 0

        return {"by_symbol": by_symbol, "by_class": by_class, "by_day": by_day}

    def _get_heatmap_data(self) -> list[dict]:
        """Feature 11: Win/loss heatmap — day of week vs hour P&L."""
        heatmap = {}  # (day, hour) -> {"pnl": 0, "count": 0}
        for t in self.closed_trades:
            try:
                dt = datetime.fromisoformat(t.closed_at)
                key = (dt.weekday(), dt.hour)
                heatmap.setdefault(key, {"pnl": 0, "count": 0})
                heatmap[key]["pnl"] += t.pnl
                heatmap[key]["count"] += 1
            except Exception:
                pass
        # Convert to flat list for frontend
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return [
            {"day": day_names[d], "hour": h, "pnl": round(v["pnl"], 2), "count": v["count"]}
            for (d, h), v in sorted(heatmap.items())
        ]

    def _get_drawdown_recovery(self) -> dict:
        """Feature 12: Track current drawdown and recovery progress."""
        if not self.closed_trades:
            return {"in_drawdown": False, "current_dd": 0, "peak": self.peak_equity,
                    "trough": self.equity, "recovery_pct": 100, "drawdown_trades": 0}
        # Walk through equity curve to find current drawdown episode
        running = self.starting_balance
        peak = running
        trough = running
        dd_start_trade = 0
        in_dd = False
        for i, t in enumerate(self.closed_trades):
            running += t.pnl
            if running > peak:
                peak = running
                trough = running
                in_dd = False
                dd_start_trade = i + 1
            elif running < trough:
                trough = running
                in_dd = True

        current_dd = peak - self.equity
        dd_pct = (current_dd / peak * 100) if peak > 0 else 0
        recovery_needed = peak - self.equity
        recovery_pct = 0
        if in_dd and peak > trough:
            recovered = self.equity - trough
            total_dd = peak - trough
            recovery_pct = round(recovered / total_dd * 100, 1) if total_dd > 0 else 0
        else:
            recovery_pct = 100

        return {
            "in_drawdown": current_dd > 10,  # >$10 is meaningful
            "current_dd": round(current_dd, 2),
            "current_dd_pct": round(dd_pct, 2),
            "peak": round(peak, 2),
            "trough": round(trough, 2),
            "recovery_pct": recovery_pct,
            "recovery_needed": round(recovery_needed, 2),
            "drawdown_trades": len(self.closed_trades) - dd_start_trade,
        }

    # ── State Snapshot (for API) ────────────────────────────────────────────
    def _build_equity_curve(self) -> list[dict]:
        """Build equity curve from closed trades for charting."""
        curve = [{"time": datetime.now().isoformat(), "balance": self.starting_balance}]
        running = self.starting_balance
        for t in self.closed_trades:
            running += t.pnl
            curve.append({"time": t.closed_at, "balance": round(running, 2)})
        return curve

    def get_state(self) -> dict:
        self._update_equity()
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "unrealized_pnl": round(self.equity - self.balance, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "hard_floor": self.hard_floor,
            "buffer_to_floor": round(self.equity - self.hard_floor, 2),
            "peak_equity": round(self.peak_equity, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(win_rate, 1),
            "open_positions": [p.to_dict() for p in self.positions.values()],
            "closed_trades": [t.to_dict() for t in self.closed_trades[-30:]],
            "prices": {k: round(v, 4) for k, v in self.prices.items()},
            "indicators": {
                k: {ik: round(iv, 4) if isinstance(iv, float) else iv for ik, iv in v.items()}
                for k, v in self.indicators.items()
            },
            "price_history": {sym: list(hist)[-80:] for sym, hist in self.price_history.items()},
            "htf_indicators": {
                k: {ik: round(iv, 4) if isinstance(iv, float) else iv for ik, iv in v.items()}
                for k, v in self.htf_indicators.items()
            },
            "session": self._get_current_session(),
            "equity_curve": self._build_equity_curve(),
            "symbol_params": self.symbol_params,
            "equity_curve_active": len(self._recent_trades_pnl) >= 5 and sum(self._recent_trades_pnl) / len(self._recent_trades_pnl) < 0,
            "news_sentiment": self.news_sentiment,
            "ai_enabled": self.ai_brain.enabled,
            "ai_explanations": self.ai_explanations,
            "ai_status": self.ai_brain.get_status() if self.ai_brain.enabled else None,
            "performance_attribution": self._get_performance_attribution(),
            "heatmap": self._get_heatmap_data(),
            "drawdown_recovery": self._get_drawdown_recovery(),
            "consecutive_losses": self._consecutive_losses,
            "loss_breaker_until": self._loss_breaker_until,
            "vol_norm_count": sum(
                1 for sym in SYMBOLS
                if len(self._atr_history.get(sym, [])) >= 10
                and self._atr_history[sym][-1] > sorted(self._atr_history[sym])[len(self._atr_history[sym]) // 2] * 1.5
            ),
            "portfolio_heat": round(sum(
                max(0, self._convert_pnl_to_usd(
                    p.symbol,
                    (p.entry_price - p.stop_loss) * p.quantity if p.side == Side.BUY
                    else (p.stop_loss - p.entry_price) * p.quantity,
                    self.prices.get(p.symbol, p.entry_price)
                )) for p in self.positions.values()
            ), 2),
            "log": self.trade_log[-50:],
            "status": self.status,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Main Loop ───────────────────────────────────────────────────────────
    def start(self):
        self.running = True
        self.status = "RUNNING"
        self._log("Trading engine started. Balance: $25,000 | Floor: $24,000")
        self._log("Strategy: Multi-factor (Trend + Mean Reversion + Momentum)")
        self._log("Risk: 0.3% per trade, ATR-based stops, 2:1 R:R ratio")

        def price_loop():
            while self.running:
                try:
                    self.fetch_prices()
                    self.status = "RUNNING"
                except Exception as e:
                    self._log(f"Price loop error: {e}", level="ERROR")
                    self.status = "PRICE_ERROR"
                time.sleep(PRICE_FETCH_INTERVAL)

        def strategy_loop():
            # Wait for initial prices
            time.sleep(PRICE_FETCH_INTERVAL + 5)
            while self.running:
                try:
                    if self.prices:
                        self.evaluate_strategy()
                except Exception as e:
                    self._log(f"Strategy error: {e}", level="ERROR")
                time.sleep(STRATEGY_INTERVAL)

        def sentiment_loop():
            """Fetch news sentiment every 5 minutes."""
            time.sleep(45)  # initial delay — let prices load first
            while self.running:
                try:
                    if self.ai_brain.enabled:
                        self.news_sentiment = self.ai_brain.fetch_news_sentiment(SYMBOLS)
                except Exception as e:
                    self._log(f"Sentiment error: {e}", level="WARN")
                time.sleep(300)  # 5 minutes

        def htf_loop():
            """Fetch 1-hour timeframe data every 5 minutes for trend filtering."""
            time.sleep(30)  # initial delay — let prices load first
            while self.running:
                try:
                    self.fetch_htf_data()
                except Exception as e:
                    self._log(f"HTF loop error: {e}", level="WARN")
                time.sleep(300)  # 5 minutes

        def optimization_loop():
            """Run walk-forward optimization daily to update adaptive parameters.
            Disabled on cloud free tier — use /api/backtest/optimize instead.
            """
            is_cloud = bool(os.environ.get("RENDER_EXTERNAL_URL"))
            if is_cloud:
                self._log("Background optimization DISABLED on free tier (use API endpoint instead)", level="INFO")
                return  # exit immediately — don't run on free tier
            time.sleep(3600)  # wait 1 hour for startup on local
            while self.running:
                try:
                    self._run_background_optimization()
                except Exception as e:
                    self._log(f"Optimization loop error: {e}", level="WARN")
                time.sleep(86400)  # run once per day

        threading.Thread(target=price_loop, daemon=True).start()
        threading.Thread(target=strategy_loop, daemon=True).start()
        threading.Thread(target=sentiment_loop, daemon=True).start()
        threading.Thread(target=htf_loop, daemon=True).start()
        threading.Thread(target=optimization_loop, daemon=True).start()

        ai_status = "AI ENABLED (Groq)" if self.ai_brain.enabled else "AI disabled (no GROQ_API_KEY)"
        self._log(ai_status)
        self._log("Multi-timeframe: 1H trend filter active (updates every 5min)")
        if bool(os.environ.get("RENDER_EXTERNAL_URL")):
            self._log("Adaptive parameters: background optimization disabled (free tier)")
        else:
            self._log("Adaptive parameters: walk-forward optimization runs daily")

    def stop(self):
        self.running = False
        self.status = "STOPPED"
        # Close all positions
        for pid in list(self.positions.keys()):
            self._close_position(pid, "Engine shutdown - closing all positions")
        self._log("Trading engine stopped.")
