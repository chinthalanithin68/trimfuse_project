"""Temperature-sensor hot-point prediction: predict the 110 C reading from the 30 C reading (+ process data).

Only MCU-C measures the hot point in production (ts_cal2_measured_on_tester). We train there and apply the
model to MCU-A and MCU-B, which would then skip the hot insertion. Their true hot readings exist in the dataset
only so we can check the prediction.

Run: python src/ts_cal.py  -> writes reports/05_ts_cal_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import PCM, POS, load_mcu

REPORT = Path(__file__).resolve().parents[1] / "reports" / "05_ts_cal_results.md"
T_COLD, T_HOT = 30.0, 110.0
NOMINAL_SLOPE = 3.2  # counts/C, datasheet-style constant
FEATURES = {"cal1_only": ["ts_cal1_30C_counts"], "cal1_pcm": ["ts_cal1_30C_counts"] + PCM,
            "cal1_pcm_pos": ["ts_cal1_30C_counts"] + PCM + POS}
LABEL = "ts_cal2_110C_counts"


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)


def fit_predict_delta(train, test, feats, kind):
    """Model the rise (cal2 - cal1), then add cal1 back: the target is smaller and better conditioned."""
    m = make_model(kind).fit(train[feats], train[LABEL] - train["ts_cal1_30C_counts"])
    return test["ts_cal1_30C_counts"].to_numpy() + m.predict(test[feats])


def score(true, pred, slope):
    err = pred - true
    deg = np.abs(err) / slope  # counts -> degrees C using each die's true slope
    return dict(mae_counts=np.abs(err).mean(), rmse_counts=np.sqrt((err ** 2).mean()), mae_degC=deg.mean(),
                within_1C=(deg <= 1).mean(), within_2C=(deg <= 2).mean())


def run():
    df = load_mcu()
    c = df[df.ts_cal2_measured_on_tester].reset_index(drop=True)
    rows = []
    tests = {"MCU-A (never measured hot)": df[df.derivative == "MCU-A"], "MCU-B (never measured hot)": df[df.derivative == "MCU-B"]}
    # in-derivative check: train on one MCU-C wafer, test on the other
    w = sorted(c.wafer_id.unique())
    held = pd.concat([c[c.wafer_id == w[1]]])
    for tname, test in {**tests, f"MCU-C held-out wafer {w[1]}": held}.items():
        train = c[c.wafer_id == w[0]] if tname.startswith("MCU-C") else c
        slope = test["ts_slope_counts_per_C_true"].to_numpy()
        true = test[LABEL].to_numpy()
        pred = test["ts_cal1_30C_counts"].to_numpy() + NOMINAL_SLOPE * (T_HOT - T_COLD)
        rows.append(dict(test_set=tname, model="nominal slope (no model)", features="cal1", **score(true, pred, slope)))
        for fname, feats in FEATURES.items():
            for kind in ("ridge", "hgb"):
                pred = fit_predict_delta(train, test, feats, kind)
                rows.append(dict(test_set=tname, model=kind, features=fname, **score(true, pred, slope)))
    return pd.DataFrame(rows), len(c)


def main():
    res, n_c = run()
    out = [f"# Temperature-sensor hot-point prediction (trained on MCU-C, {n_c} dies, 2 wafers)\n",
           res.round(3).to_markdown(index=False),
           "\nTrue slope per die is used only to convert count errors to degrees C for scoring."]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
