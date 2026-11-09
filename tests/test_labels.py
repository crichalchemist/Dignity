import numpy as np

from data.labels import forecast_labels, risk_labels, window_meta, window_starts

BLOCK = 20
SEQ = 5


def _block(spans, types=None):
    """Return (anomaly_id, is_anomalous, anomaly_type) for a 20-row block.

    spans: list of (id, first_row, last_row_inclusive, type).
    """
    aid = np.full(BLOCK, -1, dtype=np.int64)
    typ = np.full(BLOCK, "none", dtype=object)
    for a, lo, hi, t in spans:
        aid[lo : hi + 1] = a
        typ[lo : hi + 1] = t
    return aid, aid != -1, typ


AID, IS_ANOM, TYPES = _block([(7, 12, 13, "fee_spike")])


class TestWindowLabels:
    def test_window_starts_leave_room_for_the_forecast_horizon(self):
        np.testing.assert_array_equal(window_starts(160, 100, 10), np.arange(51))

    def test_any_anomalous_row_makes_the_window_positive_by_default(self):
        y = risk_labels(IS_ANOM, window_starts(20, 5, 1), 5, overlap=0.0)
        assert y.dtype == np.int8
        assert np.flatnonzero(y).tolist() == [8, 9, 10, 11, 12, 13]

    def test_overlap_must_exceed_the_configured_fraction(self):
        # 1/5 and 2/5 anomalous
        y = risk_labels(IS_ANOM, np.array([8, 9]), 5, overlap=0.4)
        assert y.tolist() == [0, 0]  # 2/5 is not > 0.4

    def test_early_warning_labels_are_cumulative_over_the_horizon(self):
        y, mask = forecast_labels(AID, IS_ANOM, np.array([5, 6, 7, 8]), 5, 3)
        assert y.dtype == np.float32 and mask.dtype == bool
        assert y[:3].tolist() == [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert mask.tolist() == [True, True, True, False]  # t=12 already anomalous
        assert (np.diff(y, axis=1) >= 0).all()

    def test_back_to_back_anomalies_count_as_a_new_onset(self):
        aid, is_anom, _ = _block([(3, 10, 11, "fee_spike"), (4, 12, 13, "fee_spike")])
        # window start 7 -> t = 11, the last id-3 row; row 12 starts id 4
        y, mask = forecast_labels(aid, is_anom, np.array([7]), 5, 1)
        assert y.tolist() == [[1.0]]
        assert mask.tolist() == [False]

    def test_window_meta_reports_onset_and_dominant_type(self):
        off, typ = window_meta(IS_ANOM, TYPES, np.array([0, 8, 10, 13]), 5)
        assert off.dtype == np.int64
        assert off.tolist() == [-1, 4, 2, 0]
        assert typ.tolist() == ["none"] + ["fee_spike"] * 3

    def test_window_meta_picks_the_majority_type_and_first_on_ties(self):
        _, is_anom, types = _block([(1, 6, 6, "a"), (2, 7, 9, "b")])
        _, typ = window_meta(is_anom, types, np.array([5]), 5)
        assert typ.tolist() == ["b"]
        _, is_anom, types = _block([(1, 6, 7, "a"), (2, 8, 9, "b")])
        _, typ = window_meta(is_anom, types, np.array([5]), 5)
        assert typ.tolist() == ["a"]
