#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
from typing import Dict

from concretum_noise_vwap_webapp import (
    download_intraday_bars,
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

    watchlist = [s.strip().upper() for s in os.getenv("WATCHLIST", "SPY,QQQ,BITO,NVDA,TSLA,BTC-USD,ETH-USD").split(",") if s.strip()]
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
                dl = download_intraday_bars(sym)
                if not dl.empty:
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
