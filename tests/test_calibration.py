import numpy as np
import pytest

from data.stats import STAT_NAMES, calibration_stats, load_reference_stats, widen_band


class TestCalibrationStats:
    def test_iid_gaussian_returns_show_no_fat_tails_or_clustering(self):
        rng = np.random.default_rng(0)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 160 * 200)))
        volume = np.exp(rng.normal(4.6, 0.5, 160 * 200))
        s = calibration_stats(price, volume)
        assert set(s) == set(STAT_NAMES)
        assert abs(s["ret_excess_kurtosis"]) < 0.5
        assert abs(s["abs_ret_lag1_acf"]) < 0.1
        assert abs(s["log_volume_lag1_acf"]) < 0.1

    def test_band_width_grows_by_twenty_percent(self):
        assert widen_band([1.0, 2.0, 3.0]) == pytest.approx((0.8, 3.2))
        assert widen_band([-0.5, 0.5]) == pytest.approx((-0.6, 0.6))

    def test_reference_table_has_a_sourced_band_for_every_stat(self):
        table = load_reference_stats()
        assert set(table) == set(STAT_NAMES)
        for name, entry in table.items():
            assert entry["min"] < entry["max"], name
            assert len(entry["sources"]) >= 4, name  # >= 2 pairs x >= 2 windows
