"""
Automated Commodity Trading Engine
Trades: BRENT, WTI, XAGUSD (Silver), XAUUSD (Gold)
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


# ── Symbol Mapping ──────────────────────────────────────────────────────────
SYMBOLS = {
    "BRENT": {"yf": "BZ=F", "name": "Brent Crude Oil", "pip_value": 0.01, "lot_size": 10},
    "WTI":   {"yf": "CL=F", "name": "WTI Crude Oil",   "pip_value": 0.01, "lot_size": 10},
    "XAGUSD": {"yf": "SI=F", "name": "Silver (XAG/USD)", "pip_value": 0.001, "lot_size": 50},
    "XAUUSD": {"yf": "GC=F", "name": "Gold (XAU/USD)",   "pip_value": 0.01, "lot_size": 1},
}

# ── Configuration ───────────────────────────────────────────────────────────
STARTING_BALANCE = 25000.0
HARD_FLOOR = 24000.0
MAX_RISK_PER_TRADE_PCT = 0.3       # 0.3% of balance per trade (~$75)
MAX_OPEN_POSITIONS = 4
MAX_POSITIONS_PER_SYMBOL = 1
PRICE_FETCH_INTERVAL = 15          # seconds between price updates
STRATEGY_INTERVAL = 20             # seconds between strategy evaluations
MAX_DAILY_LOSS = 800.0             # stop trading if daily loss exceeds this
RISK_REWARD_RATIO = 2.0            # target 2:1 reward-to-risk


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

        self.running = False
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_equity = STARTING_BALANCE
        self.max_drawdown = 0.0
        self.status = "INITIALIZING"
        self._lock = threading.Lock()

    # ── Price Fetching (multi-source with cloud fallback) ───────────────────

    def _fetch_yfinance(self, sym: str, info: dict):
        """Primary source: Yahoo Finance via yfinance."""
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
        """Fallback 3: Free commodities API via commodities-api.com or similar."""
        # Use Yahoo Finance search as a lightweight price check
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

    def fetch_prices(self):
        """Fetch current prices using multiple sources with fallback chain."""
        import pandas as pd
        for sym, info in SYMBOLS.items():
            price = None
            hist = None
            sources = [
                ("yfinance", self._fetch_yfinance),
                ("yahoo_download", self._fetch_yahoo_download),
                ("yahoo_chart_v8", self._fetch_yahoo_chart_v8),
                ("yahoo_quote", self._fetch_marketaux),
            ]
            for source_name, fetcher in sources:
                try:
                    price, hist = fetcher(sym, info)
                    if price and price > 0:
                        self._log(f"[{sym}] price ${price:.2f} via {source_name}", level="INFO")
                        break
                except Exception as e:
                    self._log(f"{source_name} error for {sym}: {type(e).__name__}: {e}", level="WARN")
                    price, hist = None, None

            if price and price > 0:
                self.prices[sym] = price
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

        # Price momentum (rate of change over 5 periods)
        roc_5 = ((closes[-1] - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0

        current_price = float(closes[-1])

        self.indicators[symbol] = {
            "sma_10": sma_10,
            "sma_20": sma_20,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
            "atr": atr,
            "roc_5": roc_5,
            "price": current_price,
            "trend": "BULLISH" if sma_10 > sma_20 else "BEARISH",
            "bb_position": (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5,
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
        return 100 - (100 / (1 + rs))

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

    def _can_open_position(self, symbol: str) -> tuple[bool, str]:
        """Check if we're allowed to open a new position."""
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False, "Max open positions reached"

        sym_count = sum(1 for p in self.positions.values() if p.symbol == symbol)
        if sym_count >= MAX_POSITIONS_PER_SYMBOL:
            return False, f"Max positions for {symbol} reached"

        if self.daily_pnl <= -MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"

        # Pre-check: ensure opening won't risk breaching floor
        buffer = self.equity - self.hard_floor
        if buffer < 200:
            return False, "Too close to hard floor, pausing new trades"

        return True, "OK"

    def _check_floor_emergency(self):
        """Emergency close all positions if equity near hard floor."""
        if self.equity <= self.hard_floor + 100:
            self._log("EMERGENCY: Equity near hard floor! Closing all positions.", level="CRITICAL")
            for pid in list(self.positions.keys()):
                self._close_position(pid, "EMERGENCY: Hard floor protection")

    # ── Trading Strategy ────────────────────────────────────────────────────
    def evaluate_strategy(self):
        """Main strategy loop: evaluate signals and manage positions."""
        self._update_equity()
        self._check_floor_emergency()
        self._check_stops_and_targets()
        self._trail_stops()

        summaries = []
        for symbol in SYMBOLS:
            if symbol not in self.indicators or symbol not in self.prices:
                continue

            ind = self.indicators[symbol]
            price = self.prices[symbol]
            can_open, reason = self._can_open_position(symbol)

            signal = self._generate_signal(symbol, ind, price)
            sig_str = f"{signal['side'].value}(str={signal['strength']:.1f})" if signal else "NONE"
            summaries.append(f"{symbol}={sig_str}")

            if not can_open:
                if signal:
                    summaries[-1] += f"[blocked:{reason}]"
                continue

            if signal:
                self._execute_signal(symbol, signal, ind, price)

        self._log(f"Strategy scan: {' | '.join(summaries)} | Equity=${self.equity:.2f} | Buffer=${self.equity - self.hard_floor:.2f}")

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

        total = trend_score + mr_score + mom_score + macd_score

        # Standard composite: need agreement from multiple factors
        if total >= 1.5:
            return {"side": Side.BUY, "strength": total, "reasons": reasons}
        elif total <= -1.5:
            return {"side": Side.SELL, "strength": abs(total), "reasons": reasons}

        # Strong mean-reversion alone (extreme conditions)
        if abs(mr_score) >= 1.5:
            side = Side.BUY if mr_score > 0 else Side.SELL
            return {"side": side, "strength": abs(mr_score), "reasons": reasons}

        # Strong trend + MACD alignment
        if abs(trend_score) >= 1 and abs(macd_score) >= 0.5 and (trend_score * macd_score > 0):
            side = Side.BUY if trend_score > 0 else Side.SELL
            reasons.append("Trend + MACD aligned")
            return {"side": side, "strength": abs(trend_score + macd_score), "reasons": reasons}

        return None

    def _execute_signal(self, symbol: str, signal: dict, ind: dict, price: float):
        """Open a position based on the signal with ATR-based stops."""
        atr = ind["atr"]
        side = signal["side"]

        # ATR-based stop loss (1.5x ATR) and take profit (3x ATR for 2:1 R:R)
        stop_distance = atr * 1.5
        tp_distance = atr * 1.5 * RISK_REWARD_RATIO

        if side == Side.BUY:
            stop_loss = price - stop_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + stop_distance
            take_profit = price - tp_distance

        quantity = self._calculate_position_size(symbol, stop_distance)
        if quantity <= 0:
            return

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
            }
        )

    def _check_stops_and_targets(self):
        """Check all open positions for SL/TP hits."""
        for pid in list(self.positions.keys()):
            pos = self.positions.get(pid)
            if not pos:
                continue

            price = self.prices.get(pos.symbol)
            if not price:
                continue

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

    def _trail_stops(self):
        """Trail stop losses in the direction of profit to lock in gains."""
        for pos in list(self.positions.values()):
            price = self.prices.get(pos.symbol)
            if not price:
                continue
            atr = self.indicators.get(pos.symbol, {}).get("atr", 0)
            if atr <= 0:
                continue

            trail_dist = atr * 1.2  # tighter than initial stop

            if pos.side == Side.BUY:
                # Price moved up: trail stop upward
                new_sl = price - trail_dist
                if new_sl > pos.stop_loss and new_sl > pos.entry_price:
                    old_sl = pos.stop_loss
                    pos.stop_loss = round(new_sl, 4)
                    self._log(f"Trailing SL for {pos.symbol} BUY: {old_sl} -> {pos.stop_loss} (locked profit)")
            else:
                # Price moved down: trail stop downward
                new_sl = price + trail_dist
                if new_sl < pos.stop_loss and new_sl < pos.entry_price:
                    old_sl = pos.stop_loss
                    pos.stop_loss = round(new_sl, 4)
                    self._log(f"Trailing SL for {pos.symbol} SELL: {old_sl} -> {pos.stop_loss} (locked profit)")

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

        with self._lock:
            self.balance += pnl
            self.daily_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1

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

    def _update_equity(self):
        """Recalculate equity = balance + unrealized PnL."""
        unrealized = 0
        for pos in self.positions.values():
            price = self.prices.get(pos.symbol, pos.entry_price)
            if pos.side == Side.BUY:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity
            unrealized += pos.unrealized_pnl

        self.equity = self.balance + unrealized
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

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

    # ── State Snapshot (for API) ────────────────────────────────────────────
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
            "closed_trades": [t.to_dict() for t in self.closed_trades[-50:]],
            "prices": {k: round(v, 4) for k, v in self.prices.items()},
            "indicators": {
                k: {ik: round(iv, 4) if isinstance(iv, float) else iv for ik, iv in v.items()}
                for k, v in self.indicators.items()
            },
            "price_history": {sym: list(hist) for sym, hist in self.price_history.items()},
            "log": self.trade_log[-100:],
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

        threading.Thread(target=price_loop, daemon=True).start()
        threading.Thread(target=strategy_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.status = "STOPPED"
        # Close all positions
        for pid in list(self.positions.keys()):
            self._close_position(pid, "Engine shutdown - closing all positions")
        self._log("Trading engine stopped.")
