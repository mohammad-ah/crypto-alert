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
        self.assertIn("<b>", payload["text"])
        self.assertIn("&gt;UB &amp; &gt;VWAP", payload["text"])

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


if __name__ == "__main__":
    unittest.main()
