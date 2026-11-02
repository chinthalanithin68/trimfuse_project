"""Predictive single-shot trim: predict the optimal code, verify, and fall back only if the check fails.

Run: python -I src/trim_models.py  -> writes reports/02_trim_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import PCM, POS, TRIMS, load_mcu, wafer_folds

REPORT = Path(__file__).resolve().parents[1] / "reports" / "02_trim_results.md"
BASE = PCM + POS
BEST_KIND = {"bg": "ridge", "hsi": "hgb", "ldo": "ridge"}  # chosen from the results table this module writes


def feature_sets(name):
    """'base' = untrimmed reading + PCM + wafer position.
    'chained' (HSI, LDO) adds the untrimmed bandgap reading, because those circuits track the bandgap.
    Only dictionary role=feature columns are used: the trimmed bandgap value is a label in the dataset and is not allowed."""
    t = TRIMS[name]
    base = [t["untrim"]] + BASE
    chained = base + ["bg_untrim_V"] if name != "bg" else base
    return {"base": base, "chained": chained}


def make_model(kind):
    if kind == "median":
        return DummyRegressor(strategy="median")
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
    raise ValueError(kind)


def predict_codes(model, X, bits):
    return np.clip(np.round(model.predict(X)), 0, 2 ** bits - 1).astype(int)


def flow_iterations(pred, true, bits):
    """Tester flow: apply predicted code and verify (1 iteration).
    Off by one: one local step (2 total). Further off: full binary search after the failed check (1 + bits)."""
    err = np.abs(pred - true)
    return np.where(err == 0, 1, np.where(err <= 1, 2, 1 + bits))


def oof_predictions(df, name, feats, kind):
    """Out-of-fold predictions with wafer-grouped folds (no wafer in both train and test)."""
    t = TRIMS[name]
    pred = np.zeros(len(df), dtype=int)
    for tr, te in wafer_folds(df):
        m = make_model(kind).fit(df.iloc[tr][feats], df.iloc[tr][t["code"]])
        pred[te] = predict_codes(m, df.iloc[te][feats], t["bits"])
    return pred


def evaluate(df, name, feats, kind):
    t = TRIMS[name]
    true = df[t["code"]].to_numpy()
    pred = oof_predictions(df, name, feats, kind)
    err = np.abs(pred - true)
    it = flow_iterations(pred, true, t["bits"])
    conv = df[t["iters"]].to_numpy()
    return dict(trim=name, model=kind, features=len(feats), exact=(err == 0).mean(), within1=(err <= 1).mean(),
                mae_codes=err.mean(), conv_iters=conv.mean(), ai_iters=it.mean(),
                conv_ms=(conv * t["ms"]).mean(), ai_ms=(it * t["ms"]).mean())


def run(derivative="MCU-A"):
    df = load_mcu()
    df = df[df.derivative == derivative].reset_index(drop=True)
    rows = []
    for name in TRIMS:
        for fs_name, feats in feature_sets(name).items():
            if fs_name == "chained" and name == "bg":
                continue
            for kind in ("median", "ridge", "hgb"):
                r = evaluate(df, name, feats, kind)
                r["feature_set"] = fs_name
                rows.append(r)
    return pd.DataFrame(rows), df


def main():
    res, df = run()
    cols = ["trim", "feature_set", "model", "exact", "within1", "mae_codes", "conv_iters", "ai_iters", "conv_ms", "ai_ms"]
    res = res[cols]
    best = res[res.model != "median"].sort_values("ai_ms").groupby("trim").first()
    conv_total = best.conv_ms.sum()
    ai_total = best.ai_ms.sum()
    out = ["# Predictive trim results (MCU-A, wafer-grouped 5-fold, out-of-fold)\n",
           res.round(3).to_markdown(index=False),
           "\n## Best model per trim (lowest verified-flow time)\n", best.round(3).to_markdown(),
           f"\nTotal trim time per die: conventional {conv_total:.1f} ms -> predictive {ai_total:.1f} ms "
           f"({1 - ai_total / conv_total:.0%} saved). Dies: {len(df)}, wafers: {df.wafer_id.nunique()}."]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
