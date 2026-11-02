"""Drift-aware trim targets and transfer learning from the base derivative (MCU-A) to MCU-C.

Packaging shifts the trimmed values between wafer sort and final test. A model of that shift lets sort
trim to (target - predicted shift) so the part lands centred after assembly.

Run: python src/drift.py  -> writes reports/03_drift_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import POS, load_mcu, wafer_folds

REPORT = Path(__file__).resolve().parents[1] / "reports" / "03_drift_results.md"

# value at sort, value at final, target, spec half-width, shift units, to-absolute conversion
PARAMS = {
    "bg": dict(sort="bg_trimmed_sort_V", fin="bg_final_V", target=1.2, tol=5e-3, scale=1e3, unit="mV"),
    "hsi": dict(sort="hsi_trimmed_sort_MHz", fin="hsi_final_MHz", target=16.0, tol=0.16, scale=100 / 16, unit="% of 16 MHz"),
    "ldo": dict(sort="ldo_trimmed_sort_V", fin="ldo_final_V", target=1.1, tol=0.022, scale=1e3, unit="mV"),
}
FEATS = POS + ["r2"]
N_TUNE = (0, 10, 25, 50, 100)


def add_shift(df):
    """Add <trim>_shift (sort->final, in the trim's own units) and r2 (radius squared, mm^2)."""
    df = df.copy()
    for k, p in PARAMS.items():
        df[f"{k}_shift"] = (df[p["fin"]] - df[p["sort"]]) * p["scale"]
    df["r2"] = df["radius_mm"] ** 2
    return df


def shift_model(kind="ridge"):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
    raise ValueError(kind)


def pass_rate(df, k, adj=None):
    """Share of dies whose final value is inside spec. adj = predicted shift (same units as <k>_shift):
    sort would have been trimmed `adj` lower, so the final value moves by the same amount."""
    p = PARAMS[k]
    fin = df[p["fin"]].to_numpy()
    if adj is not None:
        fin = fin - np.asarray(adj) / p["scale"]
    return float((np.abs(fin - p["target"]) <= p["tol"]).mean())


def within_derivative(a):
    """Wafer-grouped out-of-fold drift model on one derivative."""
    rows = []
    for k in PARAMS:
        y = a[f"{k}_shift"].to_numpy()
        for kind in ("ridge", "hgb"):
            oof = np.zeros(len(a))
            for tr, te in wafer_folds(a):
                oof[te] = shift_model(kind).fit(a.iloc[tr][FEATS], y[tr]).predict(a.iloc[te][FEATS])
            rows.append(dict(trim=k, model=kind, unit=PARAMS[k]["unit"], mae_naive=np.abs(y - y.mean()).mean(),
                             mae_model=np.abs(y - oof).mean(),
                             pass_conv=pass_rate(a, k), pass_drift_aware=pass_rate(a, k, oof)))
    return pd.DataFrame(rows)


def transfer_curve(a, c, k, seeds=20):
    """Tune on n dies of MCU-C wafer 1, test on all of wafer 2 (tune and test wafers differ).
    Methods: zero-shot (MCU-A model), mean_offset (constant = mean shift of the n dies), scale-correct (y = a*pred_A + b fitted on n dies),
    C-only (ridge on n dies), pooled (A + n weighted C dies with is_c flag)."""
    wafers = sorted(c.wafer_id.unique())
    out = []
    base = shift_model("ridge").fit(a[FEATS], a[f"{k}_shift"])
    for tune_w, test_w in [(wafers[0], wafers[1]), (wafers[1], wafers[0])]:
        pool, test = c[c.wafer_id == tune_w], c[c.wafer_id == test_w]
        yt = test[f"{k}_shift"].to_numpy()
        pa_test = base.predict(test[FEATS])
        for n in N_TUNE:
            for seed in range(seeds if n else 1):
                res = {"zero_shot": pa_test}
                if n:
                    tune = pool.sample(n, random_state=seed)
                    yn = tune[f"{k}_shift"].to_numpy()
                    cal = LinearRegression().fit(base.predict(tune[FEATS]).reshape(-1, 1), yn)
                    res["scale_correct"] = cal.predict(pa_test.reshape(-1, 1))
                    res["c_only"] = shift_model("ridge").fit(tune[FEATS], yn).predict(test[FEATS]) if n >= 10 else pa_test
                    X = pd.concat([a[FEATS].assign(is_c=0), tune[FEATS].assign(is_c=1)])
                    y = np.r_[a[f"{k}_shift"].to_numpy(), yn]
                    w = np.r_[np.ones(len(a)), np.full(n, len(a) / n * 0.2)]
                    pm = shift_model("hgb").fit(X, y, sample_weight=w)
                    res["pooled_hgb"] = pm.predict(test[FEATS].assign(is_c=1))
                    res["mean_offset"] = np.full(len(test), yn.mean())
                for m, p in res.items():
                    out.append(dict(trim=k, n_tune=n, method=m, mae=np.abs(yt - p).mean(),
                                    pass_final=pass_rate(test, k, p)))
    df = pd.DataFrame(out)
    return df.groupby(["trim", "n_tune", "method"], as_index=False)[["mae", "pass_final"]].mean()


def main():
    df = add_shift(load_mcu())
    # latent defects add extra shift but are unknown at sort; they stay in (screened by the anomaly module)
    a, b, c = (df[df.derivative == d] for d in ("MCU-A", "MCU-B", "MCU-C"))
    out = ["# Drift-aware targets and transfer learning\n"]

    out += ["## 1. Drift model inside MCU-A (wafer-grouped out-of-fold)\n",
            within_derivative(a.reset_index(drop=True)).round(4).to_markdown(index=False)]

    base = {k: shift_model("ridge").fit(a[FEATS], a[f"{k}_shift"]) for k in PARAMS}
    rows = []
    for name, d in (("MCU-B", b), ("MCU-C", c)):
        for k in PARAMS:
            p = base[k].predict(d[FEATS])
            y = d[f"{k}_shift"].to_numpy()
            rows.append(dict(derivative=name, trim=k, unit=PARAMS[k]["unit"], mae_zero_shot=np.abs(y - p).mean(),
                             mae_naive_mean_of_A=np.abs(y - a[f"{k}_shift"].mean()).mean(),
                             pass_conv=pass_rate(d, k), pass_zero_shot=pass_rate(d, k, p)))
    out += ["\n## 2. Zero-shot: MCU-A model applied to MCU-B and MCU-C\n", pd.DataFrame(rows).round(4).to_markdown(index=False)]

    c = c.reset_index(drop=True)
    curves = pd.concat([transfer_curve(a, c, k) for k in PARAMS])
    out += ["\n## 3. Transfer to MCU-C: tune on n dies of one wafer, test on the other (both directions, 20 draws)\n"]
    for k in PARAMS:
        piv = curves[curves.trim == k].pivot(index="n_tune", columns="method", values="mae")
        out += [f"\n### {k} - MAE of shift ({PARAMS[k]['unit']})\n", piv.round(3).to_markdown()]
        piv = curves[curves.trim == k].pivot(index="n_tune", columns="method", values="pass_final")
        out += [f"\n### {k} - final-spec pass rate with drift-aware target\n", piv.round(4).to_markdown()]
    out.append(f"\nConventional (trim to nominal) MCU-C final-spec pass: "
               + ", ".join(f"{k} {pass_rate(c, k):.1%}" for k in PARAMS))
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
