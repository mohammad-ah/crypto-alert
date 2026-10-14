import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

import concretum_noise_vwap_webapp as webapp
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

    def test_crypto_24h_session_and_fractional_sizing(self):
        for sym in ("BTC-USD", "ETH-USD"):
            sig, bt_df = evaluate_concretum_noise_vwap(sym, self.basket[sym], account_equity=100_000.0)
            self.assertGreater(sig.upper_bound, sig.lower_bound)
            # 24/7 session: weekend days must be part of the backtest
            self.assertTrue(any(pd.Timestamp(d).dayofweek >= 5 for d in bt_df["day"]))
            # whole-share rounding would force at least 1 BTC (~$63k) regardless of the vol target
            self.assertNotEqual(sig.trade_shares, int(sig.trade_shares))

    def test_telegram_payload_is_html_and_failures_are_reported(self):
        sig, _ = evaluate_concretum_noise_vwap("SPY", self.basket["SPY"])
        with mock.patch.object(webapp.requests, "post") as post:
            post.return_value.ok = False
            post.return_value.status_code = 400
            post.return_value.json.return_value = {"description": "Bad Request"}
            ok, status = webapp.send_concretum_alert(sig, "TOKEN", "CHAT")
        payload = post.call_args.kwargs["json"]
        self.assertFalse(ok)
        self.assertIn("Bad Request", status)
        self.assertEqual(payload["parse_mode"], "HTML")
        self.assertIn("<b>KEEP YOUR SPY LONG</b>", payload["text"])
        self.assertIn(f"<code>${sig.active_stop:,.2f}</code>", payload["text"])
        self.assertEqual(
            webapp._markdown_to_telegram_html("*A & B* <x> `σ_open`"),
            "<b>A &amp; B</b> &lt;x&gt; <code>σ_open</code>",
        )

    def test_alert_text_for_crypto_and_wait(self):
        sig, _ = evaluate_concretum_noise_vwap("ETH-USD", self.basket["ETH-USD"], account_equity=100_000.0)
        text = webapp.format_concretum_alert(sig)
        self.assertEqual(sig.action_on_30m_clock, "FLAT_WAIT")
        self.assertTrue(text.startswith("⚪ *ETH-USD: NO TRADE, WAIT*"))
        self.assertNotIn("09:30", text)
        self.assertNotIn("UTC", text)
        # 00:00 UTC is 7 PM CDT in summer and 6 PM CST in winter
        self.assertRegex(text, r"session open \((7:00 PM CDT|6:00 PM CST)\)")

    def test_alert_times_follow_daylight_saving(self):
        summer = webapp._alert_time(pd.Timestamp("2026-07-01 00:00", tz="UTC"))
        winter = webapp._alert_time(pd.Timestamp("2026-12-01 00:00", tz="UTC"))
        nyse_close = webapp._alert_time(pd.Timestamp("2026-12-01 16:00", tz="America/New_York"))
        self.assertEqual((summer, winter, nyse_close), ("7:00 PM CDT", "6:00 PM CST", "3:00 PM CST"))

    def test_app_stays_locked_without_correct_password(self):
        from streamlit.testing.v1 import AppTest

        app_path = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
        with mock.patch.object(webapp, "_app_password", return_value=""):
            at = AppTest.from_file(app_path).run()
            self.assertIn("locked", at.error[0].value)
            self.assertEqual(len(at.subheader), 0)

        with mock.patch.object(webapp, "_app_password", return_value="correct-horse"):
            at = AppTest.from_file(app_path).run()
            at.text_input[0].input("wrong-guess")
            at.button[0].click()
            at.run()
            self.assertEqual(at.error[0].value, "Incorrect password.")
            self.assertEqual(len(at.subheader), 0)

    def test_push_button_uses_saved_secrets_without_exposing_them(self):
        from streamlit.testing.v1 import AppTest

        saved = {
            "TELEGRAM_BOT_TOKEN": "111:saved-token",
            "TELEGRAM_CHAT_ID": "4242",
            "DISCORD_WEBHOOK_URL": "https://discord.example/webhook",
        }
        app_path = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
        with mock.patch.object(webapp, "_config_secret", side_effect=lambda name: saved.get(name, "")), \
                mock.patch.object(webapp, "download_intraday_bars", return_value=pd.DataFrame()), \
                mock.patch.object(webapp.requests, "post") as post:
            post.return_value.ok = True
            at = AppTest.from_file(app_path)
            at.session_state["authenticated"] = True
            at.run()
            alert_boxes = [t for t in at.text_input if t.label in ("Telegram Bot Token", "Telegram Chat ID", "Discord Webhook URL")]
            self.assertEqual([t.value for t in alert_boxes], ["", "", ""])
            at.button(key="btn_SPY").click()
            at.run()

        urls = [c.args[0] for c in post.call_args_list]
        self.assertIn("https://api.telegram.org/bot111:saved-token/sendMessage", urls)
        self.assertIn("https://discord.example/webhook", urls)
        tg_call = next(c for c in post.call_args_list if "telegram" in c.args[0])
        self.assertEqual(tg_call.kwargs["json"]["chat_id"], "4242")
        self.assertEqual(at.success[0].value, "Telegram OK | Discord OK")

    def test_completed_decision_bar_keeps_enter_and_exit_after_later_bars(self):
        eth = self.basket["ETH-USD"]
        after_enter = eth[eth.index <= pd.Timestamp("2026-10-13 03:35:00")]
        after_exit = eth[eth.index <= pd.Timestamp("2026-10-13 04:05:00")]

        off_enter, _ = evaluate_concretum_noise_vwap("ETH-USD", after_enter, use_completed_decision_bar=False)
        on_enter, _ = evaluate_concretum_noise_vwap("ETH-USD", after_enter, use_completed_decision_bar=True)
        self.assertEqual(off_enter.action_on_30m_clock, "HOLD_LONG")
        self.assertEqual(on_enter.action_on_30m_clock, "ENTER_LONG")

        off_exit, _ = evaluate_concretum_noise_vwap("ETH-USD", after_exit, use_completed_decision_bar=False)
        on_exit, _ = evaluate_concretum_noise_vwap("ETH-USD", after_exit, use_completed_decision_bar=True)
        self.assertEqual(off_exit.action_on_30m_clock, "FLAT_WAIT")
        self.assertEqual(on_exit.action_on_30m_clock, "EXIT_LONG_STOP")

        full_off, _ = evaluate_concretum_noise_vwap("ETH-USD", eth, use_completed_decision_bar=False)
        full_on, _ = evaluate_concretum_noise_vwap("ETH-USD", eth, use_completed_decision_bar=True)
        self.assertEqual(full_off.action_on_30m_clock, "FLAT_WAIT")
        self.assertEqual(full_on.action_on_30m_clock, "FLAT_WAIT")

    def test_alerts_stay_per_ticker_and_flag_changes_actionable_rules(self):
        from scanner_cron import ticker_alert_is_actionable, ticker_alert_state

        spy, _ = evaluate_concretum_noise_vwap("SPY", self.basket["SPY"])
        eth, _ = evaluate_concretum_noise_vwap("ETH-USD", self.basket["ETH-USD"])
        self.assertNotEqual(spy.ticker, eth.ticker)
        self.assertNotEqual(ticker_alert_state(spy, False), ticker_alert_state(eth, False))
        self.assertNotEqual(ticker_alert_state(spy, True), ticker_alert_state(eth, True))

        self.assertEqual(eth.action_on_30m_clock, "FLAT_WAIT")
        self.assertFalse(ticker_alert_is_actionable(eth, True), "completed-check mode must not spam WAIT")
        self.assertTrue("HOLD" in spy.action_on_30m_clock)
        self.assertTrue(ticker_alert_is_actionable(spy, True), "completed-check mode alerts HOLD so a moved stop is sent")
        self.assertTrue(ticker_alert_is_actionable(spy, False), "legacy mode still alerts IMBALANCE/HOLD preview")


if __name__ == "__main__":
    unittest.main()
