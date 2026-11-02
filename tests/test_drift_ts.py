import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_mcu  # noqa: E402
from drift import FEATS, PARAMS, add_shift, pass_rate, shift_model  # noqa: E402
from ts_cal import FEATURES, LABEL, fit_predict_delta, score  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return add_shift(load_mcu())


def test_shift_units(df):
    d = df.iloc[0]
    assert d.bg_shift == pytest.approx((d.bg_final_V - d.bg_trimmed_sort_V) * 1000)
    assert d.hsi_shift == pytest.approx((d.hsi_final_MHz / d.hsi_trimmed_sort_MHz - 1) * 100, abs=1e-3)


def test_pass_rate_adjustment_moves_final_value(df):
    c = df[df.derivative == "MCU-C"]
    base = pass_rate(c, "bg")
    perfect = pass_rate(c, "bg", c["bg_shift"])  # knowing the true shift cannot be worse
    assert perfect >= base
    assert 0.0 <= base <= 1.0


def test_pass_rate_zero_adjustment_equals_no_adjustment(df):
    a = df[df.derivative == "MCU-A"]
    assert pass_rate(a, "ldo", np.zeros(len(a))) == pass_rate(a, "ldo")


def test_mcu_c_drifts_more_than_a(df):
    m = df.groupby("derivative").bg_shift.mean()
    assert m["MCU-C"] > m["MCU-A"] > m["MCU-B"]


def test_drift_model_fits_without_labels_as_features(df):
    assert not {p["fin"] for p in PARAMS.values()} & set(FEATS)
    m = shift_model("ridge").fit(df[FEATS], df["bg_shift"])
    assert np.isfinite(m.predict(df[FEATS])).all()


def test_ts_features_do_not_include_label(df):
    for feats in FEATURES.values():
        assert LABEL not in feats


def test_ts_pcm_model_generalises_to_other_derivative(df):
    c = df[df.ts_cal2_measured_on_tester]
    a = df[df.derivative == "MCU-A"]
    pred = fit_predict_delta(c, a, FEATURES["cal1_pcm"], "ridge")
    s = score(a[LABEL].to_numpy(), pred, a["ts_slope_counts_per_C_true"].to_numpy())
    assert s["within_2C"] > 0.9
