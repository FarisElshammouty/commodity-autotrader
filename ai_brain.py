"""
AI Brain Module — Groq-powered intelligence layer for the trading bot.
Uses Groq free tier (llama-3.3-70b-versatile, 30 RPM).
Gracefully degrades to no-op when API key is missing.
"""

import os
import json
import time
import threading
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime

try:
    import requests as _requests
except ImportError:
    _requests = None


class AIBrain:
    """Central AI module: trade signals, sentiment analysis, trade explanations."""

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"
    MAX_RPM = 25  # 30 limit, 5 headroom

    # Google News RSS for commodity news (no API key needed)
    NEWS_RSS = (
        "https://news.google.com/rss/search?"
        "q=%22crude+oil%22+OR+%22gold+price%22+OR+%22silver+price%22+OR+%22brent%22+OR+%22commodity%22"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    NEWS_RSS_FALLBACK = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?"
        "s=GC=F,CL=F,SI=F,BZ=F&region=US&lang=en-US"
    )

    def __init__(self, log_callback=None):
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.enabled = bool(self.api_key)
        self._log_fn = log_callback or (lambda msg, level="INFO": print(f"[AI-{level}] {msg}"))
        self._lock = threading.Lock()
        self._call_timestamps: deque = deque(maxlen=60)
        self._sentiment_cache: dict = {}       # {symbol: {"score": float, "summary": str, "updated": str}}
        self._last_indicators: dict = {}       # {symbol: hash} — track meaningful changes
        self._total_calls = 0
        self._total_errors = 0

    # ── Rate Limiter ──────────────────────────────────────────────────────

    def _rate_limit_ok(self) -> bool:
        """Check sliding-window rate limit (MAX_RPM calls per 60s)."""
        now = time.time()
        with self._lock:
            # Remove timestamps older than 60s
            while self._call_timestamps and now - self._call_timestamps[0] > 60:
                self._call_timestamps.popleft()
            return len(self._call_timestamps) < self.MAX_RPM

    def _record_call(self):
        with self._lock:
            self._call_timestamps.append(time.time())
            self._total_calls += 1

    @property
    def calls_last_minute(self) -> int:
        now = time.time()
        return sum(1 for t in self._call_timestamps if now - t < 60)

    # ── Core Groq API Call ────────────────────────────────────────────────

    def _call_groq(self, messages: list, max_tokens: int = 300, temperature: float = 0.3) -> str | None:
        """Make a single Groq API call. Returns response text or None on failure."""
        if not self.enabled or not _requests:
            return None
        if not self._rate_limit_ok():
            self._log_fn("Rate limit reached, skipping AI call", "WARN")
            return None

        try:
            resp = _requests.post(
                self.GROQ_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=10,
            )
            self._record_call()

            if resp.status_code == 429:
                self._log_fn("Groq rate limited (429), backing off", "WARN")
                return None
            if resp.status_code != 200:
                self._log_fn(f"Groq error {resp.status_code}: {resp.text[:200]}", "WARN")
                self._total_errors += 1
                return None

            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except _requests.exceptions.Timeout:
            self._log_fn("Groq API timeout (10s)", "WARN")
            self._total_errors += 1
            return None
        except Exception as e:
            self._log_fn(f"Groq API error: {type(e).__name__}: {e}", "WARN")
            self._total_errors += 1
            return None

    # ── Feature 1: AI Trade Signal Enhancement ────────────────────────────

    def should_call_ai_signal(self, symbol: str, composite_score: float, indicators: dict) -> bool:
        """Gate: only call AI when the signal is in the uncertain zone and indicators changed."""
        if not self.enabled:
            return False

        # Only for uncertain scores (algo isn't highly confident)
        abs_score = abs(composite_score)
        if abs_score < 0.5 or abs_score > 3.5:
            return False  # Too weak (no signal) or too strong (algo confident enough)

        # Check if indicators changed meaningfully since last call
        rsi = indicators.get("rsi", 50)
        trend = indicators.get("trend", "")
        bb_pos = indicators.get("bb_position", 0.5)
        ind_hash = f"{trend}_{round(rsi, 0)}_{round(bb_pos, 1)}"

        prev_hash = self._last_indicators.get(symbol)
        if prev_hash == ind_hash:
            return False  # Nothing meaningful changed

        self._last_indicators[symbol] = ind_hash
        return True

    def enhance_signal(
        self,
        symbol: str,
        symbol_name: str,
        composite_score: float,
        signal_side: str,
        indicators: dict,
        price_history: list,
        signal_reasons: list,
    ) -> dict:
        """
        Feature 1: Ask AI to evaluate the trade signal.
        Returns {"action": "BUY"/"SELL"/"HOLD", "confidence": 0-100, "reasoning": str, "ai_used": bool}
        """
        default = {"action": signal_side, "confidence": 0, "reasoning": "", "ai_used": False}
        if not self.enabled:
            return default

        # Build concise price summary (last 15 points)
        prices_summary = ""
        recent = price_history[-15:] if len(price_history) > 15 else price_history
        if recent:
            prices_summary = ", ".join(f"{p['price']:.2f}" for p in recent)

        system_msg = (
            "You are a professional commodity trader. Analyze the data and respond with ONLY valid JSON. "
            "No markdown, no explanation outside JSON."
        )
        user_msg = (
            f"Symbol: {symbol_name} ({symbol})\n"
            f"Current Price: ${indicators.get('price', 0):.2f}\n"
            f"Recent Prices (oldest→newest): [{prices_summary}]\n\n"
            f"Indicators:\n"
            f"  RSI(14): {indicators.get('rsi', 0):.1f}\n"
            f"  SMA10: {indicators.get('sma_10', 0):.2f}, SMA20: {indicators.get('sma_20', 0):.2f}\n"
            f"  MACD Line: {indicators.get('macd_line', 0):.4f}, Signal: {indicators.get('macd_signal', 0):.4f}, Hist: {indicators.get('macd_histogram', 0):.4f}\n"
            f"  Bollinger Bands: Upper={indicators.get('bb_upper', 0):.2f}, Mid={indicators.get('bb_mid', 0):.2f}, Lower={indicators.get('bb_lower', 0):.2f}, Position={indicators.get('bb_position', 0):.2f}\n"
            f"  ATR: {indicators.get('atr', 0):.4f}, ROC(5): {indicators.get('roc_5', 0):.2f}%\n"
            f"  Trend: {indicators.get('trend', 'UNKNOWN')}\n\n"
            f"Algorithm Signal: {signal_side} (composite score: {composite_score:+.2f})\n"
            f"Algorithm Reasons: {'; '.join(signal_reasons)}\n\n"
            f"Should this trade be taken? Respond in JSON:\n"
            f'{{"action": "BUY" or "SELL" or "HOLD", "confidence": 0-100, "reasoning": "one sentence"}}'
        )

        raw = self._call_groq(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=150,
            temperature=0.2,
        )
        if not raw:
            return default

        try:
            # Clean potential markdown wrapping
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            action = result.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            confidence = min(100, max(0, int(result.get("confidence", 50))))
            reasoning = str(result.get("reasoning", ""))[:200]

            self._log_fn(
                f"[AI Signal] {symbol}: {action} ({confidence}%) — {reasoning}",
                "INFO",
            )
            return {"action": action, "confidence": confidence, "reasoning": reasoning, "ai_used": True}

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._log_fn(f"AI signal parse error: {e}, raw: {raw[:100]}", "WARN")
            return default

    # ── Feature 2: News Sentiment Analysis ────────────────────────────────

    def fetch_news_sentiment(self, symbols: dict) -> dict:
        """
        Feature 2: Fetch commodity news and analyze sentiment per symbol.
        Returns {symbol: {"score": float, "summary": str, "updated": str}}
        """
        if not self.enabled or not _requests:
            return self._sentiment_cache

        # Fetch news headlines from Google News RSS
        headlines = self._fetch_news_headlines()
        if not headlines:
            return self._sentiment_cache  # Keep cached values

        # Build prompt for sentiment analysis
        system_msg = (
            "You are a financial news analyst. Analyze headlines for commodity market sentiment. "
            "Respond with ONLY valid JSON, no markdown."
        )
        headlines_text = "\n".join(f"- {h}" for h in headlines[:12])
        symbol_names = {s: info.get("name", s) for s, info in symbols.items()}

        user_msg = (
            f"Recent commodity news headlines:\n{headlines_text}\n\n"
            f"Analyze sentiment for these commodities:\n"
            f"- BRENT ({symbol_names.get('BRENT', 'Brent Crude')})\n"
            f"- WTI ({symbol_names.get('WTI', 'WTI Crude')})\n"
            f"- XAGUSD ({symbol_names.get('XAGUSD', 'Silver')})\n"
            f"- XAUUSD ({symbol_names.get('XAUUSD', 'Gold')})\n\n"
            f"Respond with JSON: {{\"BRENT\": {{\"score\": -1.0 to 1.0, \"summary\": \"brief\"}}, "
            f"\"WTI\": ..., \"XAGUSD\": ..., \"XAUUSD\": ...}}\n"
            f"Score: -1.0=very bearish, 0=neutral, +1.0=very bullish"
        )

        raw = self._call_groq(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=300,
            temperature=0.2,
        )
        if not raw:
            return self._sentiment_cache

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            now = datetime.now().isoformat()

            for sym in ["BRENT", "WTI", "XAGUSD", "XAUUSD"]:
                if sym in result and isinstance(result[sym], dict):
                    score = float(result[sym].get("score", 0))
                    score = max(-1.0, min(1.0, score))
                    summary = str(result[sym].get("summary", ""))[:150]
                    self._sentiment_cache[sym] = {
                        "score": round(score, 2),
                        "summary": summary,
                        "updated": now,
                    }

            parts = [f"{s}={v.get('score', 0):+.2f}" for s, v in self._sentiment_cache.items()]
            self._log_fn(f"Sentiment updated: {', '.join(parts)}", "INFO")
            return self._sentiment_cache

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._log_fn(f"Sentiment parse error: {e}", "WARN")
            return self._sentiment_cache

    def _fetch_news_headlines(self) -> list:
        """Fetch recent commodity news headlines from Google News RSS."""
        if not _requests:
            return []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        for url in [self.NEWS_RSS, self.NEWS_RSS_FALLBACK]:
            try:
                resp = _requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.text)
                # RSS format: <rss><channel><item><title>...</title></item></channel></rss>
                headlines = []
                for item in root.iter("item"):
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        # Clean up Google News title format: "Headline - Source"
                        title = title_el.text.strip()
                        headlines.append(title)
                    if len(headlines) >= 15:
                        break
                if headlines:
                    return headlines
            except Exception as e:
                self._log_fn(f"News fetch error ({url[:40]}...): {e}", "WARN")
                continue

        return []

    def get_sentiment(self, symbol: str) -> dict:
        """Get cached sentiment for a symbol."""
        return self._sentiment_cache.get(symbol, {"score": 0.0, "summary": "No data", "updated": ""})

    # ── Feature 3: AI Trade Explanations ──────────────────────────────────

    def explain_trade_open(
        self,
        symbol: str,
        symbol_name: str,
        side: str,
        price: float,
        indicators: dict,
        signal_reasons: list,
        ai_signal: dict | None = None,
    ) -> str:
        """Feature 3A: Generate AI explanation for opening a trade."""
        if not self.enabled:
            return ""

        sentiment = self.get_sentiment(symbol)
        sentiment_str = f"News sentiment: {sentiment['score']:+.2f} ({sentiment['summary']})" if sentiment["summary"] != "No data" else "No news sentiment data"

        system_msg = "You are a trading analyst. Write a clear, concise 2-3 sentence explanation. No jargon. Be specific about the numbers."
        user_msg = (
            f"Explain why this trade was opened:\n\n"
            f"Action: {side} {symbol_name} ({symbol}) at ${price:.2f}\n"
            f"Indicators: RSI={indicators.get('rsi', 0):.1f}, "
            f"SMA10={indicators.get('sma_10', 0):.2f}, SMA20={indicators.get('sma_20', 0):.2f}, "
            f"MACD Hist={indicators.get('macd_histogram', 0):.4f}, "
            f"BB Position={indicators.get('bb_position', 0):.2f}, "
            f"ATR={indicators.get('atr', 0):.4f}, Trend={indicators.get('trend', 'N/A')}\n"
            f"Signal Reasons: {'; '.join(signal_reasons)}\n"
            f"{sentiment_str}\n"
        )
        if ai_signal and ai_signal.get("ai_used"):
            user_msg += f"AI Assessment: {ai_signal['action']} ({ai_signal['confidence']}% confidence) — {ai_signal['reasoning']}\n"

        user_msg += "\nWrite 2-3 sentences explaining the trade rationale in plain English:"

        raw = self._call_groq(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=200,
            temperature=0.3,
        )
        return raw.strip() if raw else ""

    def explain_trade_close(
        self,
        symbol: str,
        symbol_name: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason_close: str,
        duration: str = "",
    ) -> str:
        """Feature 3B: Generate AI explanation for closing a trade."""
        if not self.enabled:
            return ""

        system_msg = "You are a trading analyst. Write a clear, concise 2-3 sentence explanation. Be specific about what happened."
        user_msg = (
            f"Explain this closed trade:\n\n"
            f"Position: {side} {symbol_name} ({symbol})\n"
            f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}\n"
            f"P&L: {'+'if pnl >= 0 else ''}${pnl:.2f}\n"
            f"Close Reason: {reason_close}\n"
            f"Duration: {duration}\n\n"
            f"Write 2-3 sentences explaining what happened:"
        )

        raw = self._call_groq(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=200,
            temperature=0.3,
        )
        return raw.strip() if raw else ""

    # ── Status ────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.MODEL,
            "calls_last_minute": self.calls_last_minute,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "sentiment_symbols": list(self._sentiment_cache.keys()),
        }
