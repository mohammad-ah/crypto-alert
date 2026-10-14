"""
Concretum Intraday Noise Area + VWAP Signal Web App & Multi-Ticker Scanner
==========================================================================
Exact 1:1 Python + Streamlit implementation of `concretum_noise_area_vwap_strategy.pine`
based on:
  "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
  by Carlo Zarattini, Andrew Aziz, and Prof. Andrea Barbon (St. Gallen / SFI, 2024–2025).
  Paper FAQ Q12 proves that running this exact strategy across a basket of liquid ETFs
  (SPY, QQQ, BITO/IBIT) and top liquid US stocks (NVDA, TSLA, AMD, META, AAPL, MSFT)
  achieves a portfolio Sharpe Ratio > 2.0.

Crypto tickers in Yahoo format (BTC-USD, ETH-USD) use a 24h UTC session instead of 09:30-16:00 ET.

HOW TO RUN LOCALLY OR HOST FOR FREE ON STREAMLIT COMMUNITY CLOUD:
    pip install streamlit pandas numpy yfinance requests pytz
    streamlit run concretum_noise_vwap_webapp.py
"""

from __future__ import annotations

import datetime as dt
import hmac
import html
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import streamlit as st
except ImportError:
    st = None


@dataclass(frozen=True)
class SessionSpec:
    tz: str                      # timezone the bar index is converted to before stripping tz info
    open_minute: int             # minutes after local midnight when the session opens
    length_minutes: int
    periods_per_year: int        # trading days per year, used to annualize the backtest Sharpe
    fractional_units: bool


US_EQUITY_RTH = SessionSpec("America/New_York", 9 * 60 + 30, 390, 252, False)
CRYPTO_24H = SessionSpec("UTC", 0, 1440, 365, True)


def session_for_ticker(ticker: str) -> SessionSpec:
    return CRYPTO_24H if ticker.upper().endswith("-USD") else US_EQUITY_RTH


def download_intraday_bars(ticker: str, period: str = "30d", interval: str = "5m") -> pd.DataFrame:
    """Downloads Yahoo Finance bars with lowercase columns and a tz-naive index in the ticker's session timezone."""
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    dl = yf.download(ticker, period=period, interval=interval, progress=False)
    if dl.empty:
        return dl
    dl.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in dl.columns]
    dl.index = dl.index.tz_convert(session_for_ticker(ticker).tz).tz_localize(None)
    return dl


@dataclass
class ConcretumSignal:
    ticker: str
    intraday_regime: str         # "LONG_ACTIVE", "SHORT_ACTIVE", "EXIT_LONG_STOP", "EXIT_SHORT_STOP", "INSIDE_NOISE_AREA"
    action_on_30m_clock: str     # "ENTER_LONG", "ENTER_SHORT", "HOLD_LONG", "HOLD_SHORT", "EXIT_STOP", "FLAT_WAIT"
    price: float
    session_open: float
    prev_rth_close: float
    upper_bound: float
    lower_bound: float
    session_vwap: float
    active_stop: float
    sigma_open_pct: float
    daily_vol_14d_pct: float
    effective_leverage: float
    trade_shares: float          # whole shares for equities, fractional units for crypto
    prev_rsi5: float
    gamma_regime: str
    daily_pattern: str
    backtest_total_return_pct: float
    backtest_sharpe: float
    backtest_hit_ratio_pct: float
    backtest_mdd_pct: float
    reasons: List[str]


def _markdown_to_telegram_html(msg: str) -> str:
    # Telegram's legacy Markdown rejects the whole message on any unpaired `_` (e.g. "σ_open", "HOLD_LONG")
    text = html.escape(msg, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*([^*]+)\*", r"<b>\1</b>", text)


def send_concretum_alert(
    sig: ConcretumSignal,
    telegram_bot_token: str = "",
    telegram_chat_id: str = "",
    discord_webhook_url: str = "",
) -> Tuple[bool, str]:
    """Sends a formatted Concretum Noise Area + VWAP alert to Telegram and/or Discord."""
    if requests is None:
        return False, "Python `requests` library is not installed."

    bullets = "\n".join([f"  • {r}" for r in sig.reasons])
    msg = (
        f"⚡ *CONCRETUM NOISE + VWAP SIGNAL: {sig.ticker} — {sig.action_on_30m_clock}*\n"
        f"*Intraday Regime:* `{sig.intraday_regime}`\n"
        f"*Price:* `${sig.price:,.2f}` (09:30 Open: `${sig.session_open:,.2f}` | Prior Close: `${sig.prev_rth_close:,.2f}`)\n"
        f"*Noise Area Bounds:* Upper `${sig.upper_bound:,.2f}` | Lower `${sig.lower_bound:,.2f}` (σ_open: `{sig.sigma_open_pct:.2f}%`)\n"
        f"*Session VWAP:* `${sig.session_vwap:,.2f}` | *Active Trailing Stop:* `${sig.active_stop:,.2f}`\n"
        f"*Vol-Targeted Sizing:* `{sig.effective_leverage:.2f}x` (`{sig.trade_shares}` shares @ 14d Vol `{sig.daily_vol_14d_pct:.2f}%`)\n"
        f"*Dealer Gamma (Prior RSI-5):* `{sig.prev_rsi5:.1f}` ({sig.gamma_regime})\n"
        f"*Daily Compression Pattern:* `{sig.daily_pattern}`\n"
        f"*Why This Trade Makes Sense:*\n{bullets}\n"
        f"*Backtest Summary:* Net Return `{sig.backtest_total_return_pct:+.1f}%` | "
        f"Sharpe `{sig.backtest_sharpe:.2f}` | Daily Hit Rate `{sig.backtest_hit_ratio_pct:.0f}%` | MDD `{sig.backtest_mdd_pct:.1f}%`"
    )

    results = []
    all_ok = True
    if telegram_bot_token and telegram_chat_id:
        url = f"https://api.telegram.org/bot{telegram_bot_token.strip()}/sendMessage"
        r = requests.post(
            url,
            json={"chat_id": telegram_chat_id.strip(), "text": _markdown_to_telegram_html(msg), "parse_mode": "HTML"},
            timeout=10,
        )
        if r.ok:
            results.append("Telegram OK")
        else:
            all_ok = False
            try:
                detail = r.json().get("description", "")
            except ValueError:
                detail = r.text[:200]
            results.append(f"Telegram HTTP {r.status_code}: {detail}")

    if discord_webhook_url:
        r = requests.post(discord_webhook_url, json={"content": msg.replace("*", "**")}, timeout=10)
        if r.ok:
            results.append("Discord OK")
        else:
            all_ok = False
            results.append(f"Discord HTTP {r.status_code}: {r.text[:200]}")

    if not results:
        return False, "No Telegram or Discord webhook configured."
    return all_ok, " | ".join(results)


def evaluate_concretum_noise_vwap(
    ticker: str,
    intra_df: pd.DataFrame,
    account_equity: float = 100_000.0,
    lookback_days: int = 14,
    vol_mult: float = 1.0,
    use_gap_adjust: bool = True,
    stop_mode: str = "Curr.Band + VWAP",
    trade_freq_min: int = 30,
    sizing_mode: str = "Volatility Target (Paper Dynamic)",
    target_daily_vol: float = 0.02,
    vol_lookback: int = 14,
    max_leverage: float = 4.0,
    user_bias: str = "Both (Long & Short)",
    nr4_boost_mult: float = 1.0,
    filter_mon_early: bool = False,
    commission_per_share: float = 0.0035,
    session: Optional[SessionSpec] = None,
) -> Tuple[ConcretumSignal, pd.DataFrame]:
    """
    Exact 1:1 port of `concretum_noise_area_vwap_strategy.pine`.
    Computes the 2D [day, minute_from_open] Noise Area matrix, gap-adjusted UB/LB,
    RTH session VWAP, 30-min decision grid, dynamic trailing stop, 14d sample std
    volatility targeting, Dealer Gamma RSI(5), and NR4/NR7/Inside/Triangle patterns.

    `intra_df` must be indexed in the session timezone (see `download_intraday_bars`).
    If `session` is omitted it is inferred from the ticker (crypto `-USD` pairs -> 24h UTC).
    """
    session = session or session_for_ticker(ticker)
    sess_len = session.length_minutes
    df = intra_df.copy()
    df["day"] = df.index.date
    # Calculate bar close minute from session open (assuming 5-min bars if index is bar open)
    step_mins = 5
    if len(df) > 1:
        diffs = pd.Series(df.index).diff().dt.total_seconds().div(60).dropna()
        if not diffs.empty:
            step_mins = int(max(1, min(60, round(diffs.median()))))

    df["min_from_open"] = (df.index.hour * 60 + df.index.minute) - session.open_minute
    df = df[(df["min_from_open"] >= 0) & (df["min_from_open"] <= sess_len)].copy()
    df["bar_close_min"] = df["min_from_open"] + step_mins

    # 1. Daily stats strictly from completed prior daily bars [1..vol_lookback] (zero lookahead)
    daily_open = df.groupby("day")["open"].first()
    daily_high = df.groupby("day")["high"].max()
    daily_low = df.groupby("day")["low"].min()
    daily_close = df.groupby("day")["close"].last()

    d_ret = daily_close.pct_change()
    # Sample standard deviation over vol_lookback days (ddof=1, matching paper Eq. page 15: 1/13)
    daily_vol = d_ret.rolling(vol_lookback, min_periods=max(3, vol_lookback // 2)).std(ddof=1).shift(1)

    # Wilder 5-day RSI on daily closes, shifted by 1 day
    delta = daily_close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 5, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 5, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    daily_rsi5 = (100.0 - (100.0 / (1.0 + rs))).shift(1).fillna(50.0)

    # Daily compression patterns (NR4, NR7, Inside Day, Triangle) shifted by 1 day
    d_range = daily_high - daily_low
    is_nr4 = (d_range == d_range.rolling(4, min_periods=4).min()).shift(1, fill_value=False)
    is_nr7 = (d_range == d_range.rolling(7, min_periods=7).min()).shift(1, fill_value=False)
    is_inside = ((daily_high < daily_high.shift(1)) & (daily_low > daily_low.shift(1))).shift(1, fill_value=False)
    is_triangle = (
        (daily_high < daily_high.shift(1))
        & (daily_high < daily_high.shift(2))
        & (daily_low > daily_low.shift(1))
        & (daily_low > daily_low.shift(2))
    ).shift(1, fill_value=False)

    # 2. Exact Time-of-Day Sigma_Open Matrix & Session VWAP
    df["sess_open"] = df["day"].map(daily_open)
    df["prev_rth_close"] = df["day"].map(daily_close.shift(1))
    df["move_open"] = (df["close"] / df["sess_open"] - 1.0).abs()

    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    df["pv"] = hlc3 * df["volume"]
    df["cum_pv"] = df.groupby("day")["pv"].cumsum()
    df["cum_vol"] = df.groupby("day")["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = df["cum_pv"] / df["cum_vol"]

    df["sigma_open"] = (
        df.groupby("min_from_open")["move_open"]
        .transform(lambda s: s.rolling(lookback_days, min_periods=max(2, lookback_days // 3)).mean().shift(1))
    )

    if use_gap_adjust:
        anchor_high = np.maximum(df["sess_open"], df["prev_rth_close"].fillna(df["sess_open"]))
        anchor_low = np.minimum(df["sess_open"], df["prev_rth_close"].fillna(df["sess_open"]))
    else:
        anchor_high = df["sess_open"]
        anchor_low = df["sess_open"]

    df["UB"] = anchor_high * (1.0 + vol_mult * df["sigma_open"])
    df["LB"] = anchor_low * (1.0 - vol_mult * df["sigma_open"])

    # 3. Backtest & State Simulation Across All Days (Matching Pine Script Order Logic)
    allow_long = user_bias in ("Both (Long & Short)", "Long Only")
    allow_short = user_bias in ("Both (Long & Short)", "Short Only")

    aum = account_equity
    all_days = sorted(df["day"].unique())
    bt_rows = []
    last_day_pos = 0
    last_day_action = "FLAT_WAIT"

    for d in all_days:
        day_df = df[df["day"] == d]
        if day_df["sigma_open"].isna().all():
            continue

        s_open = float(day_df["sess_open"].iloc[0])
        d_vol = float(daily_vol.get(d, np.nan))
        raw_lev = (
            min(max_leverage, target_daily_vol / d_vol)
            if (sizing_mode == "Volatility Target (Paper Dynamic)" and pd.notna(d_vol) and d_vol > 0)
            else 1.0
        )
        pat_boost = nr4_boost_mult if (bool(is_nr4.get(d, False)) or bool(is_nr7.get(d, False))) else 1.0
        eff_lev = min(max_leverage, raw_lev * pat_boost)
        if session.fractional_units:
            shares = round((aum * eff_lev) / s_open, 4)
        else:
            shares = max(1, round((aum * eff_lev) / s_open))

        pos = 0
        trades_cnt = 0
        gross_pnl = 0.0
        closes = day_df["close"].values
        ubs = day_df["UB"].values
        lbs = day_df["LB"].values
        vwaps = day_df["vwap"].values
        b_mins = day_df["bar_close_min"].values
        is_monday = pd.Timestamp(d).dayofweek == 0

        for i in range(len(day_df)):
            if i > 0:
                gross_pnl += pos * (closes[i] - closes[i - 1]) * shares

            c_px = closes[i]
            ub_i = ubs[i]
            lb_i = lbs[i]
            vw_i = vwaps[i]
            b_min = int(b_mins[i])

            if pd.isna(ub_i) or pd.isna(lb_i) or pd.isna(vw_i):
                continue

            l_stop = max(ub_i, vw_i) if stop_mode == "Curr.Band + VWAP" else (vw_i if stop_mode == "VWAP Only" else lb_i)
            s_stop = min(lb_i, vw_i) if stop_mode == "Curr.Band + VWAP" else (vw_i if stop_mode == "VWAP Only" else ub_i)

            l_sig = (c_px > ub_i) and (c_px > vw_i)
            s_sig = (c_px < lb_i) and (c_px < vw_i)

            is_eod = b_min >= (sess_len - step_mins)
            is_decision = (b_min >= trade_freq_min) and (b_min < sess_len) and (b_min % trade_freq_min == 0)
            mon_skip = filter_mon_early and is_monday and (b_min < 60)

            action_now = "HOLD_LONG" if pos > 0 else ("HOLD_SHORT" if pos < 0 else "FLAT_WAIT")

            if is_eod and pos != 0 and d != all_days[-1]:
                trades_cnt += 1
                pos = 0
                action_now = "FLAT_EOD"
            elif is_decision and not mon_skip:
                if pos > 0:
                    if s_sig and allow_short:
                        pos = -1
                        trades_cnt += 2
                        action_now = "REVERSE_SHORT"
                    elif c_px <= l_stop:
                        pos = 0
                        trades_cnt += 1
                        action_now = "EXIT_LONG_STOP"
                elif pos < 0:
                    if l_sig and allow_long:
                        pos = 1
                        trades_cnt += 2
                        action_now = "REVERSE_LONG"
                    elif c_px >= s_stop:
                        pos = 0
                        trades_cnt += 1
                        action_now = "EXIT_SHORT_STOP"
                elif pos == 0:
                    if l_sig and allow_long:
                        pos = 1
                        trades_cnt += 1
                        action_now = "ENTER_LONG"
                    elif s_sig and allow_short:
                        pos = -1
                        trades_cnt += 1
                        action_now = "ENTER_SHORT"

            if d == all_days[-1] and i == len(day_df) - 1:
                last_day_pos = pos
                last_day_action = action_now

        comm = trades_cnt * max(0.35, commission_per_share * shares)
        net_pnl = gross_pnl - comm
        ret_d = net_pnl / aum
        aum += net_pnl
        bt_rows.append({"day": d, "aum": aum, "ret": ret_d, "trades": trades_cnt})

    bt_df = pd.DataFrame(bt_rows)
    if not bt_df.empty and len(bt_df) > 2:
        tot_ret = (bt_df["aum"].iloc[-1] / account_equity - 1.0) * 100.0
        std_r = bt_df["ret"].std(ddof=1)
        sharpe = (bt_df["ret"].mean() / std_r * math.sqrt(session.periods_per_year)) if std_r > 0 else 0.0
        traded = bt_df[bt_df["trades"] > 0]
        hit_ratio = ((traded["ret"] > 0).mean() * 100.0) if not traded.empty else 0.0
        mdd = ((bt_df["aum"] / bt_df["aum"].cummax() - 1.0).min() * -100.0)
    else:
        tot_ret, sharpe, hit_ratio, mdd = 0.0, 0.0, 0.0, 0.0

    # 4. Extract Current Bar HUD Metrics (Matching Pine Script HUD Table)
    last_bar = df.iloc[-1]
    cur_day = last_bar["day"]
    px = float(last_bar["close"])
    s_open = float(last_bar["sess_open"])
    p_close = float(last_bar["prev_rth_close"]) if pd.notna(last_bar["prev_rth_close"]) else s_open
    ub = float(last_bar["UB"]) if pd.notna(last_bar["UB"]) else px * 1.004
    lb = float(last_bar["LB"]) if pd.notna(last_bar["LB"]) else px * 0.996
    vwap = float(last_bar["vwap"]) if pd.notna(last_bar["vwap"]) else px
    sig_open = float(last_bar["sigma_open"]) * 100.0 if pd.notna(last_bar["sigma_open"]) else 0.35

    l_stop = max(ub, vwap) if stop_mode == "Curr.Band + VWAP" else (vwap if stop_mode == "VWAP Only" else lb)
    s_stop = min(lb, vwap) if stop_mode == "Curr.Band + VWAP" else (vwap if stop_mode == "VWAP Only" else ub)
    act_stop = l_stop if (last_day_pos > 0 or (px > ub and px > vwap)) else (s_stop if (last_day_pos < 0 or (px < lb and px < vwap)) else vwap)

    d_vol = float(daily_vol.get(cur_day, 0.015))
    if math.isnan(d_vol) or d_vol <= 0:
        d_vol = 0.015
    raw_lev = min(max_leverage, target_daily_vol / d_vol) if sizing_mode == "Volatility Target (Paper Dynamic)" else 1.0
    nr4_flag = bool(is_nr4.get(cur_day, False))
    nr7_flag = bool(is_nr7.get(cur_day, False))
    tri_flag = bool(is_triangle.get(cur_day, False))
    ins_flag = bool(is_inside.get(cur_day, False))
    eff_lev = min(max_leverage, raw_lev * (nr4_boost_mult if (nr4_flag or nr7_flag) else 1.0))
    cur_eq = float(bt_df["aum"].iloc[-1]) if not bt_df.empty else account_equity
    if session.fractional_units:
        shares = round((cur_eq * eff_lev) / s_open, 4)
    else:
        shares = max(1, round((cur_eq * eff_lev) / s_open))

    rsi5_val = float(daily_rsi5.get(cur_day, 50.0))
    gamma_txt = (
        "NEGATIVE GAMMA (Strong Trend Edge)"
        if rsi5_val < 35
        else ("POSITIVE GAMMA (Mean-Reversion Risk)" if rsi5_val > 65 else "NEUTRAL GAMMA (Normal Trend)")
    )
    comp_txt = (
        "NR7 COMPRESSION (2.5+ Sharpe)"
        if nr7_flag
        else ("NR4 COMPRESSION (3.2 Sharpe!)" if nr4_flag else ("TRIANGLE (2.0 Sharpe)" if tri_flag else ("INSIDE DAY" if ins_flag else "Standard Range")))
    )

    if px > ub and px > vwap:
        regime_txt = "BULLISH IMBALANCE (>UB & >VWAP)"
        if last_day_action == "FLAT_WAIT" and allow_long:
            last_day_action = "ENTER_LONG (On 30m Close)"
    elif px < lb and px < vwap:
        regime_txt = "BEARISH IMBALANCE (<LB & <VWAP)"
        if last_day_action == "FLAT_WAIT" and allow_short:
            last_day_action = "ENTER_SHORT (On 30m Close)"
    else:
        regime_txt = "INSIDE NOISE AREA (Equilibrium)"

    reasons = [
        f"Noise Area Bounds (VM={vol_mult:.2f}, {lookback_days}d): UB=${ub:,.2f} | LB=${lb:,.2f} | Session VWAP=${vwap:,.2f}",
        f"Active Trailing Stop ({stop_mode}): ${act_stop:,.2f} (checked every {trade_freq_min}m at :00/:30)",
        f"Volatility Sizing: 14d Daily Vol={d_vol*100:.2f}% -> {eff_lev:.2f}x leverage ({shares} shares)",
        f"Regime Context: Prior RSI(5)={rsi5_val:.1f} [{gamma_txt}] | Pattern=[{comp_txt}]",
    ]

    sig_obj = ConcretumSignal(
        ticker=ticker,
        intraday_regime=regime_txt,
        action_on_30m_clock=last_day_action,
        price=px,
        session_open=s_open,
        prev_rth_close=p_close,
        upper_bound=ub,
        lower_bound=lb,
        session_vwap=vwap,
        active_stop=act_stop,
        sigma_open_pct=sig_open,
        daily_vol_14d_pct=d_vol * 100.0,
        effective_leverage=eff_lev,
        trade_shares=shares,
        prev_rsi5=rsi5_val,
        gamma_regime=gamma_txt,
        daily_pattern=comp_txt,
        backtest_total_return_pct=tot_ret,
        backtest_sharpe=sharpe,
        backtest_hit_ratio_pct=hit_ratio,
        backtest_mdd_pct=mdd,
        reasons=reasons,
    )
    return sig_obj, bt_df


def generate_concretum_demo_basket() -> Dict[str, pd.DataFrame]:
    """Generates realistic 5-minute bars across SPY, QQQ, BITO, NVDA, TSLA (RTH) and BTC-USD, ETH-USD (24/7 UTC) for offline testing."""
    rng = np.random.default_rng(7)
    b_days = pd.bdate_range("2026-09-08", "2026-10-13")
    specs = [
        ("SPY", 570.0, 0.00035, 0.0008),
        ("QQQ", 490.0, 0.00045, 0.0010),
        ("BITO", 22.5, 0.00060, 0.0016),
        ("NVDA", 135.0, 0.00075, 0.0018),
        ("TSLA", 250.0, -0.00055, 0.0020),
    ]
    basket = {}
    for sym, start_px, last_day_drift, bar_sigma in specs:
        rows = []
        px = start_px
        for d_idx, d in enumerate(b_days):
            px *= (1.0 + rng.normal(0.0003, 0.003))
            is_last = d_idx == len(b_days) - 1
            drift = last_day_drift if is_last else rng.choice([-0.0002, 0.00025, 0.0], p=[0.35, 0.40, 0.25])
            scale_range = 0.55 if d_idx == len(b_days) - 2 else 1.0  # Make yesterday an NR4 compression day!
            for m in range(0, 390, 5):
                ts = dt.datetime.combine(d.date(), dt.time(9, 30)) + dt.timedelta(minutes=m)
                o = px
                c = o * (1.0 + drift + rng.normal(0, bar_sigma * scale_range))
                h = max(o, c) * (1.0 + abs(rng.normal(0, bar_sigma * 0.45 * scale_range)))
                l = min(o, c) * (1.0 - abs(rng.normal(0, bar_sigma * 0.45 * scale_range)))
                v = int(rng.uniform(200_000, 750_000))
                rows.append({"dt": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
                px = c
        basket[sym] = pd.DataFrame(rows).set_index("dt")

    c_days = pd.date_range("2026-09-08", "2026-10-13", freq="D")
    crypto_specs = [
        ("BTC-USD", 63_000.0, 0.00010, 0.0015, (5.0, 60.0)),
        ("ETH-USD", 2_600.0, -0.00012, 0.0020, (200.0, 2_000.0)),
    ]
    for sym, start_px, last_day_drift, bar_sigma, (v_lo, v_hi) in crypto_specs:
        rows = []
        px = start_px
        for d_idx, d in enumerate(c_days):
            is_last = d_idx == len(c_days) - 1
            drift = last_day_drift if is_last else rng.choice([-0.00004, 0.00005, 0.0], p=[0.35, 0.40, 0.25])
            scale_range = 0.55 if d_idx == len(c_days) - 2 else 1.0
            for m in range(0, 1440, 5):
                ts = dt.datetime.combine(d.date(), dt.time(0, 0)) + dt.timedelta(minutes=m)
                o = px
                c = o * (1.0 + drift + rng.normal(0, bar_sigma * scale_range))
                h = max(o, c) * (1.0 + abs(rng.normal(0, bar_sigma * 0.45 * scale_range)))
                l = min(o, c) * (1.0 - abs(rng.normal(0, bar_sigma * 0.45 * scale_range)))
                v = float(rng.uniform(v_lo, v_hi))
                rows.append({"dt": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
                px = c
        basket[sym] = pd.DataFrame(rows).set_index("dt")
    return basket


def _app_password() -> str:
    try:
        from_secrets = st.secrets.get("APP_PASSWORD", "")
    except Exception:  # no secrets.toml configured
        from_secrets = ""
    return str(from_secrets or os.getenv("APP_PASSWORD", ""))


def _password_gate() -> bool:
    """Blocks the app until APP_PASSWORD is entered; stays locked if no password is configured."""
    if st.session_state.get("authenticated"):
        return True
    expected = _app_password()
    if not expected:
        st.error("This app is locked: set APP_PASSWORD in Streamlit secrets or the environment.")
        return False
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Unlock"):
            if hmac.compare_digest(entered.encode(), expected.encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("Incorrect password.")
    return False


def render_streamlit_app() -> None:
    st.set_page_config(page_title="Concretum Noise Area + VWAP Signal Hub", layout="wide")
    if not _password_gate():
        return
    st.title("🎯 Concretum Intraday Noise Area + VWAP Signal App")
    st.caption(
        "Exact 1:1 Python & Multi-Ticker Scanner port of `concretum_noise_area_vwap_strategy.pine` "
        "(Zarattini, Aziz, & Barbon — Univ. of St. Gallen / Swiss Finance Institute)."
    )

    with st.sidebar:
        st.header("1. Core Strategy Parameters")
        watchlist_str = st.text_input("Ticker Watchlist (comma-separated)", value="SPY,QQQ,BITO,NVDA,TSLA,BTC-USD,ETH-USD")
        equity = st.number_input("Account Capital ($)", min_value=1000.0, value=100000.0, step=5000.0)
        lookback_days = st.slider("Sigma Lookback Window (Days)", min_value=5, max_value=90, value=14)
        vol_mult = st.slider("Volatility Multiplier (VM)", min_value=0.5, max_value=2.5, value=1.0, step=0.05)
        use_gap = st.checkbox("Adjust Noise Area for Overnight Gaps", value=True)
        stop_mode = st.selectbox("Trailing Stop Formula", ["Curr.Band + VWAP", "VWAP Only", "Opposite Band"])
        trade_freq = st.selectbox("Decision & Stop Interval (Minutes)", [30, 15, 60, 5], index=0)

        st.header("2. Volatility Sizing & Context Bias")
        sizing_mode = st.selectbox("Position Sizing Method", ["Volatility Target (Paper Dynamic)", "Full Notional (100% Equity)"])
        target_vol_pct = st.slider("Target Daily Volatility (%)", min_value=0.5, max_value=6.0, value=2.0, step=0.25)
        max_lev = st.slider("Max Broker Leverage Cap (x)", min_value=1.0, max_value=4.0, value=4.0, step=0.5)
        user_bias = st.selectbox("Your Directional Bias Control", ["Both (Long & Short)", "Long Only", "Short Only"])
        nr4_boost = st.slider("NR4 / NR7 Compression Size Multiplier", min_value=1.0, max_value=2.0, value=1.25, step=0.05)
        filter_mon = st.checkbox("Skip Monday Trades Before 10:30 ET", value=False)

        st.header("3. Push Alert Webhooks")
        tg_token = st.text_input("Telegram Bot Token", type="password")
        tg_chat = st.text_input("Telegram Chat ID")
        discord_url = st.text_input("Discord Webhook URL", type="password")
        feed_mode = st.radio("Data Feed", ["Live Yahoo Finance (5m RTH)", "Built-in Verification Dataset"])

    basket = generate_concretum_demo_basket()
    tickers = [t.strip().upper() for t in watchlist_str.split(",") if t.strip()]

    if feed_mode == "Live Yahoo Finance (5m RTH)" and yf is not None:
        live_basket = {}
        for sym in tickers:
            try:
                dl = download_intraday_bars(sym)
                if not dl.empty:
                    live_basket[sym] = dl
            except Exception as exc:
                st.warning(f"Fallback to built-in data for {sym}: {exc}")
        if live_basket:
            basket = live_basket

    signals: List[ConcretumSignal] = []
    for sym, df_sym in basket.items():
        sig, _ = evaluate_concretum_noise_vwap(
            ticker=sym,
            intra_df=df_sym,
            account_equity=equity,
            lookback_days=lookback_days,
            vol_mult=vol_mult,
            use_gap_adjust=use_gap,
            stop_mode=stop_mode,
            trade_freq_min=trade_freq,
            sizing_mode=sizing_mode,
            target_daily_vol=target_vol_pct / 100.0,
            max_leverage=max_lev,
            user_bias=user_bias,
            nr4_boost_mult=nr4_boost,
            filter_mon_early=filter_mon,
        )
        signals.append(sig)

    # Display HUD Cards for Each Ticker (Identical to Pine Script HUD)
    cols = st.columns(min(3, max(1, len(signals))))
    for idx, sig in enumerate(signals):
        with cols[idx % len(cols)]:
            st.subheader(f"{sig.ticker} — {sig.action_on_30m_clock}")
            st.metric("Intraday Regime", sig.intraday_regime, f"Sharpe: {sig.backtest_sharpe:.2f}")
            st.write(f"**Price:** `${sig.price:,.2f}` | **Upper / Lower Bound:** `${sig.upper_bound:,.2f}` / `${sig.lower_bound:,.2f}`")
            st.write(f"**Session VWAP / Active Stop:** `${sig.session_vwap:,.2f}` / `${sig.active_stop:,.2f}`")
            st.write(f"**14d Daily Vol / Target Size:** `{sig.daily_vol_14d_pct:.2f}%` → `{sig.effective_leverage:.2f}x` (`{sig.trade_shares}` sh)")
            st.write(f"**Dealer Gamma (Prior RSI-5):** `{sig.prev_rsi5:.1f}` | {sig.gamma_regime}")
            st.write(f"**Daily Pattern Setup:** `{sig.daily_pattern}`")
            if st.button(f"Push {sig.ticker} Alert to Phone", key=f"btn_{sig.ticker}"):
                ok, status = send_concretum_alert(sig, tg_token, tg_chat, discord_url)
                (st.success if ok else st.error)(status)

    st.subheader("Multi-Ticker Concretum Noise Area + VWAP Scanner Table")
    st.dataframe(
        pd.DataFrame([
            {
                "Ticker": s.ticker,
                "30m Action": s.action_on_30m_clock,
                "Intraday Regime": s.intraday_regime,
                "Price ($)": round(s.price, 2),
                "Upper Bound ($)": round(s.upper_bound, 2),
                "Lower Bound ($)": round(s.lower_bound, 2),
                "VWAP ($)": round(s.session_vwap, 2),
                "Active Stop ($)": round(s.active_stop, 2),
                "14d Vol (%)": round(s.daily_vol_14d_pct, 2),
                "Target Shares": s.trade_shares,
                "Prior RSI-5 (Gamma)": round(s.prev_rsi5, 1),
                "Daily Pattern": s.daily_pattern,
            }
            for s in signals
        ]),
        use_container_width=True,
    )


if __name__ == "__main__":
    if st is not None and hasattr(st, "runtime") and st.runtime.exists():
        render_streamlit_app()
    else:
        demo_basket = generate_concretum_demo_basket()
        print("=== CONCRETUM NOISE AREA + VWAP MULTI-TICKER SCANNER VERIFICATION ===")
        for sym, df_s in demo_basket.items():
            s, _ = evaluate_concretum_noise_vwap(sym, df_s, nr4_boost_mult=1.25)
            print(
                f"[{s.ticker}] {s.action_on_30m_clock} | {s.intraday_regime} | "
                f"Px=${s.price:,.2f} | UB=${s.upper_bound:,.2f} | LB=${s.lower_bound:,.2f} | "
                f"VWAP=${s.session_vwap:,.2f} | Stop=${s.active_stop:,.2f} | "
                f"Size={s.effective_leverage:.2f}x ({s.trade_shares} sh) | {s.daily_pattern}"
            )
