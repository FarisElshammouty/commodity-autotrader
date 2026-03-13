"""
Flask Web Server for the Automated Trading Dashboard
"""

import os
import threading
import time
import requests as _req

from flask import Flask, render_template, jsonify
from trading_engine import TradingEngine

app = Flask(__name__)
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
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    return jsonify(engine.get_state())


@app.route("/api/stop", methods=["POST"])
def api_stop():
    engine.stop()
    return jsonify({"status": "stopped"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "engine": engine.status})


@app.route("/api/debug")
def api_debug():
    """Debug endpoint to diagnose cloud price fetching issues."""
    import sys
    results = {}
    sym, info = "XAUUSD", {"yf": "GC=F", "name": "Gold", "pip_value": 0.01, "lot_size": 1}
    for name, fn in [
        ("yfinance", engine._fetch_yfinance),
        ("yahoo_download", engine._fetch_yahoo_download),
        ("yahoo_chart_v8", engine._fetch_yahoo_chart_v8),
        ("yahoo_quote", engine._fetch_marketaux),
    ]:
        try:
            price, hist = engine._run_with_timeout(fn, (sym, info), timeout_sec=15)
            results[name] = {"price": price, "hist_len": len(hist) if hist is not None else 0}
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
    return jsonify({
        "python": sys.version,
        "test_symbol": "XAUUSD (GC=F)",
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
