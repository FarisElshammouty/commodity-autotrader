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


# ── Startup ────────────────────────────────────────────────────────────
engine.start()

# Start keep-alive pinger in background
threading.Thread(target=_keep_alive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print(f"  TRADING DASHBOARD: http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
