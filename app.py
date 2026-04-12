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

VERSION = "v4.0"
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
    is_cloud = bool(os.environ.get("RENDER_EXTERNAL_URL"))
    days = int(request.args.get("days", 90 if is_cloud else 180))
    days = min(max(days, 30), 180 if is_cloud else 365)  # clamp tighter on cloud
    syms = request.args.get("symbols", "").strip()
    symbol_list = [s.strip().upper() for s in syms.split(",") if s.strip()] if syms else None
    # On free tier, limit to max 5 symbols at once to avoid OOM
    if is_cloud and symbol_list is None:
        symbol_list = list(bt.SYMBOLS.keys())[:5]
    elif is_cloud and symbol_list and len(symbol_list) > 5:
        symbol_list = symbol_list[:5]
    result = bt.run_full_backtest(symbols=symbol_list, period_days=days)
    return jsonify(result)


@app.route("/api/backtest/optimize")
def api_backtest_optimize():
    """Run walk-forward optimization for a single symbol.
    Uses a reduced param grid on cloud to avoid OOM on free tier (512MB).
    """
    symbol = request.args.get("symbol", "XAUUSD").upper()
    if symbol not in bt.SYMBOLS:
        return jsonify({"error": f"Unknown symbol: {symbol}"}), 400
    is_cloud = bool(os.environ.get("RENDER_EXTERNAL_URL"))
    days = int(request.args.get("days", 90 if is_cloud else 180))
    days = min(max(days, 30), 180 if is_cloud else 365)
    # On free tier, use a reduced parameter grid to prevent OOM
    if is_cloud:
        import gc
        gc.collect()  # free memory before heavy operation
        saved_grid = bt.PARAM_GRID.copy()
        bt.PARAM_GRID = {
            "signal_threshold": [1.5, 2.5],
            "atr_stop_mult":    [2.0],
            "risk_reward":      [1.5, 2.5],
            "max_risk_pct":     [0.20],
        }
        try:
            result = bt.run_optimization(symbol, period_days=days)
        finally:
            bt.PARAM_GRID = saved_grid
            gc.collect()
    else:
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
    engine._consecutive_losses = 0
    engine._consecutive_wins = 0
    engine._best_win_streak = 0
    engine._best_loss_streak = 0
    engine._loss_breaker_until = 0
    engine._recent_trades_pnl.clear()
    engine._trade_cooldowns.clear()
    engine._loss_cooldowns.clear()
    engine._equity_snapshots.clear()
    engine._last_equity_snapshot = 0
    # Wipe database
    db.reset_db()
    engine._log("Engine reset to fresh $25,000 start", level="INFO")
    return jsonify({"status": "reset", "balance": STARTING_BALANCE})


@app.route("/api/monte-carlo")
def api_monte_carlo():
    """Run Monte Carlo drawdown analysis on closed trades."""
    # Get trade PnLs from either backtest or live trades
    source = request.args.get("source", "live")  # "live" or "backtest"
    if source == "live":
        trades = db.load_trades()
        if not trades or len(trades) < 5:
            return jsonify({"error": "Need at least 5 closed trades for Monte Carlo analysis"})
        pnls = [t["pnl"] for t in trades]
    else:
        # Run backtest first, then use those PnLs
        symbol = request.args.get("symbol", "XAUUSD").upper()
        days = int(request.args.get("days", 180))
        days = min(max(days, 90), 365)
        result = bt.run_full_backtest(symbols=[symbol], period_days=days)
        sym_result = result.get("results", {}).get(symbol, {})
        if "error" in sym_result:
            return jsonify(sym_result)
        # Extract PnLs from backtest trades
        bt_trades = sym_result.get("trades_count", 0)
        if bt_trades < 5:
            return jsonify({"error": f"Backtest produced only {bt_trades} trades"})
        # Re-run to get actual trade PnLs
        hist = bt.fetch_history(symbol, days)
        if hist is None:
            return jsonify({"error": "Could not fetch history"})
        from trading_engine import SIGNAL_THRESHOLD, ATR_STOP_MULT, RISK_REWARD_RATIO, MAX_RISK_PER_TRADE_PCT
        params = {"signal_threshold": SIGNAL_THRESHOLD, "atr_stop_mult": ATR_STOP_MULT,
                  "risk_reward": RISK_REWARD_RATIO, "max_risk_pct": MAX_RISK_PER_TRADE_PCT}
        bt_result = bt.run_backtest(symbol, hist, params)
        pnls = [t.pnl_net for t in bt_result.trades]

    n_sims = int(request.args.get("simulations", 1000))
    n_sims = min(max(n_sims, 100), 5000)
    result = bt.monte_carlo_drawdown(pnls, n_simulations=n_sims)
    return jsonify(result)


@app.route("/api/journal")
def api_journal():
    """Return recent signal journal entries."""
    limit = int(request.args.get("limit", 100))
    limit = min(max(limit, 10), 500)
    signals = db.load_signals(limit=limit)
    return jsonify({"signals": signals, "count": len(signals)})


@app.route("/api/analytics")
def api_analytics():
    """Performance attribution, heatmap, and drawdown recovery data."""
    return jsonify({
        "attribution": engine._get_performance_attribution(),
        "heatmap": engine._get_heatmap_data(),
        "drawdown": engine._get_drawdown_recovery(),
    })


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
