"""
Backtesting Engine — Tier 1 improvement.

Replays historical OHLCV data through the same signal logic used by the
live trading engine, with realistic spread and slippage modeling.
Also includes walk-forward parameter optimization per symbol.
"""

import time
import itertools
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    import requests as _requests
    _HAS_REQ = True
except ImportError:
    _HAS_REQ = False


# ── Symbol config (mirrors trading_engine.py) ───────────────────────────────
SYMBOLS = {
    "BRENT":  {"yf": "BZ=F", "name": "Brent Crude Oil",    "lot_size": 10,  "spread_pct": 0.03},
    "WTI":    {"yf": "CL=F", "name": "WTI Crude Oil",      "lot_size": 10,  "spread_pct": 0.03},
    "XAGUSD": {"yf": "SI=F", "name": "Silver (XAG/USD)",   "lot_size": 50,  "spread_pct": 0.05},
    "XAUUSD": {"yf": "GC=F", "name": "Gold (XAU/USD)",     "lot_size": 1,   "spread_pct": 0.02},
}

# Spread = realistic bid/ask cost as % of price
# Slippage = additional % cost per trade (market impact)
DEFAULT_SLIPPAGE_PCT = 0.01   # 0.01% per side
DEFAULT_SPREAD_PCT   = None   # use per-symbol value from SYMBOLS dict

# Parameter grid for walk-forward optimization
PARAM_GRID = {
    "signal_threshold": [1.5, 2.0, 2.5, 3.0],
    "atr_stop_mult":    [1.5, 2.0, 2.5],
    "risk_reward":      [1.5, 2.0, 2.5],
    "max_risk_pct":     [0.15, 0.20, 0.25],
}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BtTrade:
    symbol: str
    side: str          # BUY or SELL
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: float
    pnl: float
    pnl_net: float     # after spread + slippage
    close_reason: str  # TP, SL, TRAIL, EOD


@dataclass
class BtResult:
    symbol: str
    params: dict
    period_start: str
    period_end: str
    total_trades: int
    winning_trades: int
    win_rate: float
    gross_pnl: float
    net_pnl: float          # after costs
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float    # gross wins / gross losses
    avg_win: float
    avg_loss: float
    avg_rr_actual: float    # actual R:R achieved
    trades: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "params": self.params,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(self.win_rate, 1),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "profit_factor": round(self.profit_factor, 3),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "avg_rr_actual": round(self.avg_rr_actual, 3),
            "trades_count": len(self.trades),
        }


# ── Indicator Helpers (standalone, no engine dependency) ────────────────────

def _ema(data: np.ndarray, period: int) -> float:
    if len(data) < period:
        return float(data[-1])
    mult = 2 / (period + 1)
    ema = float(data[-period])
    for p in data[-period + 1:]:
        ema = (float(p) - ema) * mult + ema
    return ema


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(-period, 0)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001
    return 100 - (100 / (1 + avg_gain / avg_loss))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(highs) < period + 1:
        return float(highs[-1] - lows[-1]) if len(highs) > 0 else 1.0
    trs = []
    for i in range(-period, 0):
        tr = max(
            float(highs[i] - lows[i]),
            abs(float(highs[i] - closes[i - 1])),
            abs(float(lows[i] - closes[i - 1])),
        )
        trs.append(tr)
    return sum(trs) / len(trs)


def _compute_indicators(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Optional[dict]:
    if len(closes) < 26:
        return None

    sma_10 = float(closes[-10:].mean())
    sma_20 = float(closes[-20:].mean())
    ema_12 = _ema(closes, 12)
    ema_26 = _ema(closes, 26)
    macd_line = ema_12 - ema_26

    macd_signal = macd_line * 0.8
    if len(closes) >= 35:
        macd_series = []
        for i in range(34, len(closes)):
            e12 = _ema(closes[:i + 1], 12)
            e26 = _ema(closes[:i + 1], 26)
            macd_series.append(e12 - e26)
        if len(macd_series) >= 9:
            macd_signal = _ema(np.array(macd_series), 9)

    macd_histogram = macd_line - macd_signal
    rsi = _rsi(closes, 14)
    bb_mid = sma_20
    std_20 = float(closes[-20:].std())
    bb_upper = bb_mid + 2 * std_20
    bb_lower = bb_mid - 2 * std_20
    atr = _atr(highs, lows, closes, 14)
    roc_5 = float((closes[-1] - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0
    price = float(closes[-1])
    bb_pos = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

    return {
        "sma_10": sma_10, "sma_20": sma_20,
        "ema_12": ema_12, "ema_26": ema_26,
        "macd_line": macd_line, "macd_signal": macd_signal, "macd_histogram": macd_histogram,
        "rsi": rsi,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower, "bb_position": bb_pos,
        "atr": atr, "roc_5": roc_5, "price": price,
        "trend": "BULLISH" if sma_10 > sma_20 else "BEARISH",
    }


# ── Signal Generation (mirrors trading_engine._generate_signal) ─────────────

def _generate_signal(ind: dict, params: dict) -> Optional[dict]:
    reasons = []
    threshold = params["signal_threshold"]

    # Factor 1: Trend
    trend_score = 0
    if ind["sma_10"] > ind["sma_20"]:
        trend_score = 0.5
        if ind["macd_histogram"] > 0:
            trend_score = 1.0
    elif ind["sma_10"] < ind["sma_20"]:
        trend_score = -0.5
        if ind["macd_histogram"] < 0:
            trend_score = -1.0

    # Factor 2: Mean Reversion
    mr_score = 0
    if ind["rsi"] < 25 and ind["bb_position"] < 0.15:
        mr_score = 1.5
    elif ind["bb_position"] < 0.25 and ind["rsi"] < 40:
        mr_score = 1.0
    elif ind["rsi"] > 75 and ind["bb_position"] > 0.85:
        mr_score = -1.5
    elif ind["bb_position"] > 0.75 and ind["rsi"] > 60:
        mr_score = -1.0

    # Factor 3: Momentum
    mom_score = 0
    if ind["roc_5"] > 0.2:
        mom_score = 1.0
    elif ind["roc_5"] < -0.2:
        mom_score = -1.0

    # Factor 4: MACD
    macd_score = 0
    if ind["macd_histogram"] > 0 and ind["macd_line"] > 0:
        macd_score = 0.5
    elif ind["macd_histogram"] < 0 and ind["macd_line"] < 0:
        macd_score = -0.5

    total = trend_score + mr_score + mom_score + macd_score

    if total >= threshold:
        return {"side": "BUY", "strength": total, "reasons": reasons}
    elif total <= -threshold:
        return {"side": "SELL", "strength": abs(total), "reasons": reasons}
    if abs(mr_score) >= 1.5 and abs(total) >= 1.0:
        return {"side": "BUY" if mr_score > 0 else "SELL", "strength": abs(mr_score), "reasons": reasons}
    if abs(trend_score) >= 1 and abs(macd_score) >= 0.5 and (trend_score * macd_score > 0) and abs(total) >= 1.5:
        return {"side": "BUY" if trend_score > 0 else "SELL", "strength": abs(trend_score + macd_score), "reasons": reasons}
    return None


# ── Core Backtest Runner ─────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    hist: pd.DataFrame,
    params: dict,
    starting_balance: float = 25000.0,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    progress_callback=None,
) -> BtResult:
    """
    Run a single backtest for one symbol over the given historical data.

    params keys: signal_threshold, atr_stop_mult, risk_reward, max_risk_pct
    """
    info = SYMBOLS[symbol]
    spread_pct = info["spread_pct"] / 100
    slippage_pct_dec = slippage_pct / 100
    lot_size = info["lot_size"]

    closes = hist["Close"].values.astype(float)
    highs  = hist["High"].values.astype(float)
    lows   = hist["Low"].values.astype(float)
    times  = hist.index if hasattr(hist.index, "strftime") else list(range(len(closes)))

    balance = starting_balance
    peak    = starting_balance
    max_dd  = 0.0
    trades: list[BtTrade] = []
    daily_returns = []

    # Rolling position state
    open_pos = None  # {"side", "entry", "sl", "tp", "qty", "entry_time", "entry_cost"}

    warmup = 40  # bars needed for indicators

    for i in range(warmup, len(closes)):
        c = closes[:i + 1]
        h = highs[:i + 1]
        lo = lows[:i + 1]
        price = float(closes[i])
        t = str(times[i]) if not isinstance(times[i], str) else times[i]

        # Check open position first
        if open_pos:
            side  = open_pos["side"]
            sl    = open_pos["sl"]
            tp    = open_pos["tp"]
            qty   = open_pos["qty"]
            entry = open_pos["entry"]

            close_reason = None
            exit_price = price

            if side == "BUY":
                if lows[i] <= sl:
                    exit_price = sl
                    close_reason = "SL"
                elif highs[i] >= tp:
                    exit_price = tp
                    close_reason = "TP"
                else:
                    # Check trailing stop
                    atr_cur = _atr(h, lo, c, 14)
                    trail_trigger = atr_cur * params["atr_stop_mult"]
                    trail_dist = atr_cur * 1.2
                    if price - entry >= trail_trigger:
                        new_sl = price - trail_dist
                        if new_sl > open_pos["sl"] and new_sl > entry:
                            open_pos["sl"] = round(new_sl, 4)
            else:  # SELL
                if highs[i] >= sl:
                    exit_price = sl
                    close_reason = "SL"
                elif lows[i] <= tp:
                    exit_price = tp
                    close_reason = "TP"
                else:
                    atr_cur = _atr(h, lo, c, 14)
                    trail_trigger = atr_cur * params["atr_stop_mult"]
                    trail_dist = atr_cur * 1.2
                    if entry - price >= trail_trigger:
                        new_sl = price + trail_dist
                        if new_sl < open_pos["sl"] and new_sl < entry:
                            open_pos["sl"] = round(new_sl, 4)

            if close_reason:
                # Apply exit spread + slippage
                cost_pct = spread_pct + slippage_pct_dec
                if side == "BUY":
                    gross_pnl = (exit_price - entry) * qty
                    exit_cost = exit_price * cost_pct * qty
                else:
                    gross_pnl = (entry - exit_price) * qty
                    exit_cost = exit_price * cost_pct * qty

                net_pnl = gross_pnl - open_pos["entry_cost"] - exit_cost
                balance += net_pnl

                trades.append(BtTrade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    exit_price=exit_price,
                    entry_time=open_pos["entry_time"],
                    exit_time=t,
                    quantity=qty,
                    pnl=round(gross_pnl, 2),
                    pnl_net=round(net_pnl, 2),
                    close_reason=close_reason,
                ))
                open_pos = None

                if balance > peak:
                    peak = balance
                dd = peak - balance
                if dd > max_dd:
                    max_dd = dd

                daily_returns.append(net_pnl)

        # Only look for new entries if no position is open
        if open_pos is None:
            ind = _compute_indicators(c, h, lo)
            if ind is None:
                continue

            signal = _generate_signal(ind, params)
            if signal is None:
                continue

            atr = ind["atr"]
            stop_dist = atr * params["atr_stop_mult"]
            tp_dist   = stop_dist * params["risk_reward"]

            # Apply entry spread + slippage
            cost_pct  = spread_pct + slippage_pct_dec
            entry_cost = price * cost_pct

            if signal["side"] == "BUY":
                entry_price = price + (price * spread_pct)  # pay the spread to enter
                sl = entry_price - stop_dist
                tp = entry_price + tp_dist
            else:
                entry_price = price - (price * spread_pct)  # sell below mid
                sl = entry_price + stop_dist
                tp = entry_price - tp_dist

            # Position sizing
            risk_budget = balance * (params["max_risk_pct"] / 100)
            if stop_dist <= 0:
                continue
            raw_qty = risk_budget / stop_dist
            qty = max(lot_size, round(raw_qty / lot_size) * lot_size)
            # Cap at 20% of balance
            max_val = balance * 0.20
            if qty * price > max_val:
                qty = max(lot_size, int(max_val / price / lot_size) * lot_size)

            worst_loss = qty * stop_dist + entry_price * cost_pct * qty
            if balance - worst_loss < 24000:
                continue  # would breach floor

            entry_cost_total = entry_price * cost_pct * qty

            open_pos = {
                "side": signal["side"],
                "entry": entry_price,
                "sl": round(sl, 4),
                "tp": round(tp, 4),
                "qty": qty,
                "entry_time": t,
                "entry_cost": entry_cost_total,
            }

    # Close any remaining position at last price
    if open_pos and len(closes) > 0:
        exit_price = float(closes[-1])
        side = open_pos["side"]
        cost_pct = spread_pct + slippage_pct_dec
        if side == "BUY":
            gross_pnl = (exit_price - open_pos["entry"]) * open_pos["qty"]
        else:
            gross_pnl = (open_pos["entry"] - exit_price) * open_pos["qty"]
        exit_cost = exit_price * cost_pct * open_pos["qty"]
        net_pnl = gross_pnl - open_pos["entry_cost"] - exit_cost
        balance += net_pnl
        trades.append(BtTrade(
            symbol=symbol, side=side,
            entry_price=open_pos["entry"], exit_price=exit_price,
            entry_time=open_pos["entry_time"], exit_time=str(times[-1]),
            quantity=open_pos["qty"], pnl=round(gross_pnl, 2),
            pnl_net=round(net_pnl, 2), close_reason="EOD",
        ))
        daily_returns.append(net_pnl)

    # ── Compute Summary Stats ────────────────────────────────────────────────
    total = len(trades)
    winners = [t for t in trades if t.pnl_net > 0]
    losers  = [t for t in trades if t.pnl_net <= 0]
    win_rate = len(winners) / total * 100 if total > 0 else 0
    gross_pnl = sum(t.pnl for t in trades)
    net_pnl = sum(t.pnl_net for t in trades)
    avg_win = float(np.mean([t.pnl_net for t in winners])) if winners else 0
    avg_loss = float(np.mean([t.pnl_net for t in losers])) if losers else 0

    gross_wins  = sum(t.pnl for t in winners)
    gross_losses = abs(sum(t.pnl for t in losers))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    actual_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Sharpe ratio (annualized, assuming daily returns from trades)
    if len(daily_returns) > 1:
        dr = np.array(daily_returns)
        sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 0 else 0
    else:
        sharpe = 0.0

    period_start = str(times[warmup]) if len(times) > warmup else ""
    period_end   = str(times[-1]) if len(times) > 0 else ""

    return BtResult(
        symbol=symbol,
        params=params,
        period_start=period_start,
        period_end=period_end,
        total_trades=total,
        winning_trades=len(winners),
        win_rate=win_rate,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_rr_actual=actual_rr,
        trades=trades,
    )


# ── Historical Data Fetcher ──────────────────────────────────────────────────

def fetch_history(symbol: str, period_days: int = 180) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data. Tries yfinance first, then Yahoo chart API."""
    info = SYMBOLS[symbol]
    yf_sym = info["yf"]

    # Try yfinance (local)
    if _HAS_YF:
        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period=f"{period_days}d", interval="1h")
            if not hist.empty and len(hist) >= 50:
                return hist[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass

    # Try Yahoo chart API (cloud-friendly)
    if _HAS_REQ:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
            params = {"range": "6mo", "interval": "1h"}
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = _requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    ts = result[0].get("timestamp", [])
                    q = result[0].get("indicators", {}).get("quote", [{}])[0]
                    closes = q.get("close", [])
                    highs  = q.get("high", [])
                    lows   = q.get("low", [])
                    opens  = q.get("open", [])
                    vols   = q.get("volume", [])
                    min_len = min(len(ts), len(closes), len(highs), len(lows))
                    if min_len >= 50:
                        idx = pd.to_datetime([datetime.fromtimestamp(t) for t in ts[:min_len]])
                        hist = pd.DataFrame({
                            "Open":   opens[:min_len],
                            "High":   highs[:min_len],
                            "Low":    lows[:min_len],
                            "Close":  closes[:min_len],
                            "Volume": vols[:min_len] if len(vols) >= min_len else [0]*min_len,
                        }, index=idx).dropna()
                        return hist
        except Exception:
            pass

    return None


# ── Walk-Forward Optimization ────────────────────────────────────────────────

def walk_forward_optimize(
    symbol: str,
    hist: pd.DataFrame,
    train_pct: float = 0.7,
    progress_callback=None,
) -> dict:
    """
    Split history into train/test windows, find best params on train,
    validate on test. Returns best params + in-sample and out-of-sample results.
    """
    n = len(hist)
    train_n = int(n * train_pct)
    train_hist = hist.iloc[:train_n]
    test_hist  = hist.iloc[train_n:]

    if len(train_hist) < 60 or len(test_hist) < 20:
        return {"error": "Not enough data for walk-forward optimization"}

    # Grid search on training window
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    best_result = None
    best_score  = float("-inf")
    best_params = None
    all_train_results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if progress_callback:
            progress_callback(idx + 1, total, "train")
        try:
            result = run_backtest(symbol, train_hist, params)
            all_train_results.append(result.to_dict())
            # Score = net_pnl * profit_factor (rewards both profit and consistency)
            score = result.net_pnl * max(result.profit_factor, 0)
            if score > best_score:
                best_score  = score
                best_result = result
                best_params = params
        except Exception:
            continue

    if best_params is None:
        return {"error": "All parameter combinations failed"}

    # Validate best params on out-of-sample test window
    test_result = run_backtest(symbol, test_hist, best_params)

    return {
        "symbol": symbol,
        "best_params": best_params,
        "train_result": best_result.to_dict() if best_result else None,
        "test_result": test_result.to_dict(),
        "train_bars": len(train_hist),
        "test_bars": len(test_hist),
        "combos_tested": total,
        "all_train_summary": sorted(all_train_results, key=lambda x: x["net_pnl"], reverse=True)[:10],
    }


# ── Public API ───────────────────────────────────────────────────────────────

def run_full_backtest(symbols: list = None, period_days: int = 180) -> dict:
    """
    Run backtest for all (or specified) symbols using current live parameters.
    Returns results dict keyed by symbol.
    """
    from trading_engine import (
        SIGNAL_THRESHOLD, ATR_STOP_MULT, RISK_REWARD_RATIO, MAX_RISK_PER_TRADE_PCT
    )
    live_params = {
        "signal_threshold": SIGNAL_THRESHOLD,
        "atr_stop_mult":    ATR_STOP_MULT,
        "risk_reward":      RISK_REWARD_RATIO,
        "max_risk_pct":     MAX_RISK_PER_TRADE_PCT,
    }

    if symbols is None:
        symbols = list(SYMBOLS.keys())

    results = {}
    for sym in symbols:
        hist = fetch_history(sym, period_days)
        if hist is None or len(hist) < 60:
            results[sym] = {"error": f"Could not fetch enough history for {sym}"}
            continue
        try:
            result = run_backtest(sym, hist, live_params)
            results[sym] = result.to_dict()
        except Exception as e:
            results[sym] = {"error": str(e)}

    return {
        "params_used": live_params,
        "period_days": period_days,
        "generated_at": datetime.now().isoformat(),
        "results": results,
    }


def run_optimization(symbol: str, period_days: int = 180) -> dict:
    """Run walk-forward optimization for a single symbol."""
    hist = fetch_history(symbol, period_days)
    if hist is None or len(hist) < 80:
        return {"error": f"Not enough historical data for {symbol} ({len(hist) if hist is not None else 0} bars)"}
    return walk_forward_optimize(symbol, hist)
