"""
Flask Web Server for the Automated Trading Dashboard
"""

import os
import threading
import time
import requests as _req

from flask import Flask, render_template, jsonify, request
from trading_engine import TradingEngine, STARTING_BALANCE, HARD_FLOOR
import db
import backtest as bt

app = Flask(__name__)

VERSION = "v2.0"
engine = TradingEngine()
_engine_started = False
_start_lock = threading.Lock()


def _ensure_engine():
    """Start engine + keep-alive on first use (survives gunicorn fork)."""
    global _engine_started
    if _engine_started:
        return
    with _start_lock:
        if _engine_started:
            return
        engine.start()
        threading.Thread(target=_keep_alive, daemon=True).start()
        _engine_started = True


# ── Keep-alive self-ping (prevents Render free tier from sleeping) ─────
def _keep_alive():
    """Ping our own /health endpoint every 10 minutes to stay awake."""
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return  # local dev — no need
    url = url.rstrip("/") + "/health"
    while True:
        time.sleep(600)  # 10 minutes
        try:
            _req.get(url, timeout=10)
        except Exception:
            pass


@app.before_request
def before_request():
    _ensure_engine()


@app.route("/")
def dashboard():
    return render_template("dashboard.html", version=VERSION)


@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html", version=VERSION)


@app.route("/api/backtest/run")
def api_backtest_run():
    """Run backtest for all symbols using current live parameters."""
    days = int(request.args.get("days", 180))
    days = min(max(days, 30), 365)  # clamp 30–365
    syms = request.args.get("symbols", "").strip()
    symbol_list = [s.strip().upper() for s in syms.split(",") if s.strip()] if syms else None
    result = bt.run_full_backtest(symbols=symbol_list, period_days=days)
    return jsonify(result)


@app.route("/api/backtest/optimize")
def api_backtest_optimize():
    """Run walk-forward optimization for a single symbol."""
    symbol = request.args.get("symbol", "XAUUSD").upper()
    if symbol not in bt.SYMBOLS:
        return jsonify({"error": f"Unknown symbol: {symbol}"}), 400
    days = int(request.args.get("days", 180))
    days = min(max(days, 90), 365)
    result = bt.run_optimization(symbol, period_days=days)
    return jsonify(result)


@app.route("/api/state")
def api_state():
    state = engine.get_state()
    state["version"] = VERSION
    return jsonify(state)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    engine.stop()
    return jsonify({"status": "stopped"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "engine": engine.status})


@app.route("/api/ai-status")
def api_ai_status():
    return jsonify({
        "ai": engine.ai_brain.get_status(),
        "sentiment": engine.news_sentiment,
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Wipe database and reset engine to fresh $25,000 start."""
    # Close all open positions first
    for pid in list(engine.positions.keys()):
        engine._close_position(pid, "Manual reset")
    # Reset engine state
    engine.balance = STARTING_BALANCE
    engine.equity = STARTING_BALANCE
    engine.daily_pnl = 0.0
    engine.total_trades = 0
    engine.winning_trades = 0
    engine.peak_equity = STARTING_BALANCE
    engine.max_drawdown = 0.0
    engine.closed_trades.clear()
    engine.trade_log.clear()
    # Wipe database
    db.reset_db()
    engine._log("Engine reset to fresh $25,000 start", level="INFO")
    return jsonify({"status": "reset", "balance": STARTING_BALANCE})


@app.route("/api/debug")
def api_debug():
    """Debug endpoint to diagnose cloud price fetching issues."""
    import sys
    results = {}
    sym, info = "XAUUSD", {"yf": "GC=F", "name": "Gold", "pip_value": 0.01, "lot_size": 1}
    for name, fn in [
        ("yahoo_chart_simple", engine._fetch_yahoo_chart_simple),
        ("stooq", engine._fetch_stooq),
        ("yahoo_chart_v8", engine._fetch_yahoo_chart_v8),
        ("yahoo_quote", engine._fetch_marketaux),
        ("yfinance", engine._fetch_yfinance),
    ]:
        try:
            price, hist = engine._run_with_timeout(fn, (sym, info), timeout_sec=15)
            results[name] = {"price": price, "hist_len": len(hist) if hist is not None else 0}
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
    return jsonify({
        "python": sys.version,
        "test_symbol": "XAUUSD (GC=F)",
        "is_cloud": bool(engine._is_cloud),
        "sources": results,
        "current_prices": engine.prices,
        "log_tail": engine.trade_log[-20:],
        "engine_started": _engine_started,
    })


if __name__ == "__main__":
    _ensure_engine()
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print(f"  TRADING DASHBOARD: http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
