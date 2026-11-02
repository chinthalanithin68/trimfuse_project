"""Trim-code transfer from the base derivative (MCU-A) to the small derivatives (MCU-B, MCU-C).

Question: how many characterised dies does a new derivative need before predictive trim works on it?
Setup: train on all of MCU-A; for MCU-B / MCU-C tune on n dies of ONE wafer and test on the derivative's OTHER
wafers (every wafer takes a turn as the tuning wafer). n = 0 is zero-shot.

Methods: zero_shot (MCU-A model unchanged); linear_fix (MCU-A model + a*pred+b fitted on the n dies);
pooled (MCU-A plus the n dies, up-weighted, with a derivative flag); own_ridge (ridge on the n dies only).

Run: python src/transfer.py  -> writes reports/04_transfer_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data import TRIMS, load_mcu
from trim_models import BEST_KIND, feature_sets, flow_iterations, make_model, predict_codes

REPORT = Path(__file__).resolve().parents[1] / "reports" / "04_transfer_results.md"
N_TUNE = (0, 10, 25, 50, 100, 200)
SEEDS = 4
_BASE_CACHE = {}


def feats_for(name):
    return feature_sets(name)["base"]


def _fit_weighted(model, X, y, w):
    if hasattr(model, "steps"):  # Pipeline: pass weight to the last step
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def predictions(name, a, tune, test):
    """Return {method: predicted integer codes on test}."""
    t = TRIMS[name]
    feats = feats_for(name)
    kind = BEST_KIND[name]
    if name not in _BASE_CACHE:  # the MCU-A model never changes, so fit it once per trim
        _BASE_CACHE[name] = make_model(kind).fit(a[feats], a[t["code"]])
    base = _BASE_CACHE[name]
    raw = base.predict(test[feats])
    clip = lambda v: np.clip(np.round(v), 0, 2 ** t["bits"] - 1).astype(int)  # noqa: E731
    out = {"zero_shot": clip(raw)}
    if len(tune) >= 10:
        cal = LinearRegression().fit(base.predict(tune[feats]).reshape(-1, 1), tune[t["code"]])
        out["linear_fix"] = clip(cal.predict(raw.reshape(-1, 1)))
        X = pd.concat([a[feats].assign(is_new=0), tune[feats].assign(is_new=1)])
        y = np.r_[a[t["code"]].to_numpy(), tune[t["code"]].to_numpy()]
        w = np.r_[np.ones(len(a)), np.full(len(tune), len(a) / len(tune) * 0.2)]
        pooled = _fit_weighted(make_model(kind), X, y, w)
        out["pooled"] = clip(pooled.predict(test[feats].assign(is_new=1)))
        out["own_ridge"] = clip(make_model("ridge").fit(tune[feats], tune[t["code"]]).predict(test[feats]))
    return out


def run():
    df = load_mcu()
    a = df[df.derivative == "MCU-A"]
    rows = []
    for deriv in ("MCU-B", "MCU-C"):
        d = df[df.derivative == deriv]
        for tune_w in sorted(d.wafer_id.unique()):
            pool, test = d[d.wafer_id == tune_w], d[d.wafer_id != tune_w]
            for name, t in TRIMS.items():
                true = test[t["code"]].to_numpy()
                for n in N_TUNE:
                    if n > len(pool):
                        continue
                    for seed in range(SEEDS if n else 1):
                        tune = pool.sample(n, random_state=seed) if n else pool.iloc[:0]
                        for method, pred in predictions(name, a, tune, test).items():
                            err = np.abs(pred - true)
                            it = flow_iterations(pred, true, t["bits"])
                            rows.append(dict(derivative=deriv, trim=name, n_tune=n, method=method, within1=(err <= 1).mean(),
                                             exact=(err == 0).mean(), ai_ms=(it * t["ms"]).mean(),
                                             conv_ms=(test[t["iters"]] * t["ms"]).mean()))
    res = pd.DataFrame(rows).groupby(["derivative", "trim", "n_tune", "method"], as_index=False).mean(numeric_only=True)
    return res


def main():
    res = run()
    out = ["# Trim-code transfer from MCU-A to MCU-B / MCU-C\n",
           f"Tune on n dies of one wafer, test on the derivative's other wafers; every wafer is the tuning wafer once; "
           f"{SEEDS} random draws per n. Metric: share of dies within +/-1 code (higher is better).\n"]
    for deriv in ("MCU-B", "MCU-C"):
        for name in TRIMS:
            r = res[(res.derivative == deriv) & (res.trim == name)]
            out += [f"\n## {deriv} - {name}\n", r.pivot(index="n_tune", columns="method", values="within1").round(3).to_markdown()]
    tot = res.assign(ms=res.ai_ms).groupby(["derivative", "n_tune", "method"]).ms.sum().unstack("method")
    out += ["\n## Total trim time per die (ms), predictive flow, summed over bandgap + HSI + LDO\n",
            "Conventional is about 28.7 ms/die.\n", tot.round(2).to_markdown()]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
