import pathlib
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
