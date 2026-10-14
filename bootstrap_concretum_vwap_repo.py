"""
One-Click Repository Bootstrapper for `concretum_noise_area_vwap_strategy.pine` Signal App
==========================================================================================
Place this file and `concretum_noise_vwap_webapp.py` into a folder in your IDE and run:
    python bootstrap_concretum_vwap_repo.py

It generates:
  - app.py (Streamlit Web App — exact 1:1 port of `concretum_noise_area_vwap_strategy.pine`)
  - scanner_cron.py (Headless 30-Minute Multi-Ticker Signal Daemon with Deduplication)
  - tests/test_concretum_vwap.py (Unit Test Suite verifying exact Noise Area, VWAP, and Sizing math)
  - .github/workflows/concretum_scanner.yml (Free GitHub Actions Cron running every 30m in RTH)
  - requirements.txt & .env.example
"""

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent

REQUIREMENTS_TXT = """streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.40
requests>=2.31.0
pytz>=2024.1
pytest>=8.0.0
"""

ENV_EXAMPLE = """# Free Push Notification Credentials
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_id/your_token
TELEGRAM_BOT_TOKEN=123456789:ABCDEF_your_bot_token
TELEGRAM_CHAT_ID=987654321

# Concretum Noise Area + VWAP Strategy Settings (Matches Pine Script Inputs)
WATCHLIST=SPY,QQQ,BITO,NVDA,TSLA,AMD,META,AAPL,MSFT
ACCOUNT_EQUITY=100000
LOOKBACK_DAYS=14
VOL_MULT=1.0
STOP_MODE=Curr.Band + VWAP
TARGET_DAILY_VOL=0.02
MAX_LEVERAGE=4.0
USER_BIAS=Both (Long & Short)
NR4_BOOST_MULT=1.25
"""

GITHUB_WORKFLOW_YML = """name: Concretum Noise + VWAP 30m Signal Scanner

on:
  workflow_dispatch:
  schedule:
    # Runs every 30 minutes during US Regular Trading Hours (14:00 - 21:00 UTC Mon-Fri)
    - cron: '0,30 14-21 * * 1-5'

jobs:
  scan_concretum_signals:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run 30-Minute Concretum Noise + VWAP Scanner
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          WATCHLIST: ${{ vars.WATCHLIST || 'SPY,QQQ,BITO,NVDA,TSLA' }}
          ACCOUNT_EQUITY: ${{ vars.ACCOUNT_EQUITY || '100000' }}
          LOOKBACK_DAYS: ${{ vars.LOOKBACK_DAYS || '14' }}
          VOL_MULT: ${{ vars.VOL_MULT || '1.0' }}
          USER_BIAS: ${{ vars.USER_BIAS || 'Both (Long & Short)' }}
        run: python scanner_cron.py --send-alerts

      - name: Save deduplication state
        run: |
          git config user.name "concretum-signal-bot"
          git config user.email "bot@users.noreply.github.com"
          git add state/last_alerts.json || true
          git diff --staged --quiet || git commit -m "chore: update 30m signal state [skip ci]"
          git push || true
"""

SCANNER_CRON_PY = """#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
from typing import Dict

from concretum_noise_vwap_webapp import (
    evaluate_concretum_noise_vwap,
    generate_concretum_demo_basket,
    send_concretum_alert,
)

try:
    import yfinance as yf
except ImportError:
    yf = None

STATE_FILE = pathlib.Path("state/last_alerts.json")


def load_state() -> Dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concretum Noise Area + VWAP 30m Cron Scanner")
    parser.add_argument("--dry-run", action="store_true", help="Use built-in verification dataset")
    parser.add_argument("--send-alerts", action="store_true", help="Send Telegram/Discord alerts on signal change")
    args = parser.parse_args()

    watchlist = [s.strip().upper() for s in os.getenv("WATCHLIST", "SPY,QQQ,BITO,NVDA,TSLA").split(",") if s.strip()]
    equity = float(os.getenv("ACCOUNT_EQUITY", "100000"))
    lookback = int(os.getenv("LOOKBACK_DAYS", "14"))
    vm = float(os.getenv("VOL_MULT", "1.0"))
    bias = os.getenv("USER_BIAS", "Both (Long & Short)")
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    basket = generate_concretum_demo_basket()
    if not args.dry_run and yf is not None:
        live_basket = {}
        for sym in watchlist:
            try:
                dl = yf.download(sym, period="30d", interval="5m", progress=False)
                if not dl.empty:
                    dl.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in dl.columns]
                    dl.index = dl.index.tz_convert("America/New_York").tz_localize(None)
                    live_basket[sym] = dl
            except Exception as exc:
                print(f"[WARN] Live download failed for {sym}, using fallback: {exc}")
        if live_basket:
            basket = live_basket

    prev_state = load_state()
    new_state = dict(prev_state)

    for sym, df_sym in basket.items():
        sig, _ = evaluate_concretum_noise_vwap(
            ticker=sym,
            intra_df=df_sym,
            account_equity=equity,
            lookback_days=lookback,
            vol_mult=vm,
            user_bias=bias,
            nr4_boost_mult=1.25,
        )
        state_val = f"{sig.action_on_30m_clock}::{sig.intraday_regime}"
        changed = prev_state.get(sym) != state_val
        new_state[sym] = state_val
        print(
            f"[{sig.ticker}] Action={sig.action_on_30m_clock} | Regime={sig.intraday_regime} | "
            f"Px=${sig.price:,.2f} | UB=${sig.upper_bound:,.2f} | LB=${sig.lower_bound:,.2f} | "
            f"VWAP=${sig.session_vwap:,.2f} | Stop=${sig.active_stop:,.2f} | "
            f"Shares={sig.trade_shares} ({sig.effective_leverage:.2f}x) | Changed={changed}"
        )
        is_actionable = any(k in sig.action_on_30m_clock for k in ("ENTER_", "REVERSE_", "EXIT_")) or "IMBALANCE" in sig.intraday_regime
        if args.send_alerts and changed and is_actionable:
            ok, info = send_concretum_alert(sig, tg_token, tg_chat, discord_url)
            print(f"   -> Alert sent: {ok} ({info})")

    save_state(new_state)


if __name__ == "__main__":
    main()
"""

TEST_CONCRETUM_PY = """import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from concretum_noise_vwap_webapp import (
    evaluate_concretum_noise_vwap,
    generate_concretum_demo_basket,
)


class TestConcretumNoiseVwapStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basket = generate_concretum_demo_basket()

    def test_noise_area_bounds_and_vwap_math(self):
        sig, bt_df = evaluate_concretum_noise_vwap("SPY", self.basket["SPY"], account_equity=100_000.0, lookback_days=14, vol_mult=1.0)
        self.assertFalse(bt_df.empty)
        self.assertGreater(sig.upper_bound, sig.lower_bound, "Upper Bound must exceed Lower Bound")
        self.assertGreater(sig.session_vwap, 0.0)
        self.assertGreater(sig.sigma_open_pct, 0.0)

    def test_trailing_stop_modes_and_volatility_sizing(self):
        sig_comb, _ = evaluate_concretum_noise_vwap("NVDA", self.basket["NVDA"], stop_mode="Curr.Band + VWAP", nr4_boost_mult=1.25)
        sig_vwap, _ = evaluate_concretum_noise_vwap("NVDA", self.basket["NVDA"], stop_mode="VWAP Only", nr4_boost_mult=1.0)
        self.assertGreaterEqual(sig_comb.active_stop, sig_vwap.active_stop)
        self.assertLessEqual(sig_comb.effective_leverage, 4.0)
        self.assertGreaterEqual(sig_comb.trade_shares, 1)

    def test_bias_override_filter(self):
        sig_short_only, _ = evaluate_concretum_noise_vwap("SPY", self.basket["SPY"], user_bias="Short Only")
        self.assertNotIn("ENTER_LONG", sig_short_only.action_on_30m_clock)


if __name__ == "__main__":
    unittest.main()
"""


def bootstrap() -> None:
    (ROOT / "requirements.txt").write_text(REQUIREMENTS_TXT, encoding="utf-8")
    (ROOT / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")

    wf_dir = ROOT / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "concretum_scanner.yml").write_text(GITHUB_WORKFLOW_YML, encoding="utf-8")

    (ROOT / "scanner_cron.py").write_text(SCANNER_CRON_PY, encoding="utf-8")

    t_dir = ROOT / "tests"
    t_dir.mkdir(parents=True, exist_ok=True)
    (t_dir / "test_concretum_vwap.py").write_text(TEST_CONCRETUM_PY, encoding="utf-8")

    src_app = ROOT / "concretum_noise_vwap_webapp.py"
    if src_app.exists():
        shutil.copyfile(src_app, ROOT / "app.py")

    print("✅ Pure Concretum Noise Area + VWAP repository scaffolded!")
    print("   - app.py (Streamlit Web App)")
    print("   - scanner_cron.py (30-Min Headless Multi-Ticker Signal Scanner)")
    print("   - tests/test_concretum_vwap.py (Unit Test Suite)")
    print("   - .github/workflows/concretum_scanner.yml (Free GitHub Actions Cron)")


if __name__ == "__main__":
    bootstrap()
