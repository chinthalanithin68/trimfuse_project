"""Latent-defect screen: flag dies whose untrimmed readings disagree with what their process-monitor (PCM) data
and wafer position predict.

A die's untrimmed bandgap, oscillator, LDO and temperature-sensor readings are largely explained by PCM and
position. A large residual (robust z-score) means something else is wrong with the die. The residual models are
fit without labels, using wafer-grouped out-of-fold predictions, and `latent_defect` is only used to score.

Run: python src/anomaly.py  -> writes reports/08_anomaly_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import PCM, POS, TRIMS, load_mcu, wafer_folds

REPORT = Path(__file__).resolve().parents[1] / "reports" / "08_anomaly_results.md"
MEASURES = ["bg_untrim_V", "hsi_untrim_MHz", "ldo_untrim_V", "ts_cal1_30C_counts"]
FEATS = PCM + POS


def _model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)


def residual_z(df, kind="ridge", measures=MEASURES):
    """Per-derivative, wafer-grouped out-of-fold residuals scaled by the robust spread (MAD) of each measure.
    Returns DataFrame of |z| per measure."""
    z = pd.DataFrame(index=df.index, columns=measures, dtype=float)
    for _, g in df.groupby("derivative"):
        g = g.sort_index()
        for m in measures:
            oof = pd.Series(0.0, index=g.index)
            for tr, te in wafer_folds(g):
                oof.iloc[te] = _model(kind).fit(g.iloc[tr][FEATS], g.iloc[tr][m]).predict(g.iloc[te][FEATS])
            res = g[m] - oof
            mad = 1.4826 * np.median(np.abs(res - np.median(res)))
            z.loc[g.index, m] = np.abs(res - np.median(res)) / mad
    return z


def metrics(score, truth, final_fail):
    out = dict(auc=roc_auc_score(truth, score), ap=average_precision_score(truth, score))
    k = int(truth.sum())
    top = np.argsort(-score)[:k]  # flag the k highest-scoring dies
    out.update(recall_at_k=truth.to_numpy()[top].mean(), final_fail_in_top_k=final_fail.to_numpy()[top].mean())
    return out


def threshold_table(score, truth, final_fail, thresholds=(4, 6, 8, 10, 15)):
    rows = []
    for t in thresholds:
        f = score >= t
        rows.append(dict(z_threshold=t, flagged=int(f.sum()), flagged_pct=f.mean() * 100, caught=int((f & truth).sum()),
                         recall=(f & truth).sum() / truth.sum(), precision=(f & truth).sum() / max(f.sum(), 1),
                         final_fail_among_flagged=final_fail[f].mean() if f.any() else np.nan))
    return pd.DataFrame(rows)


def main():
    df = load_mcu().reset_index(drop=True)
    truth = df.latent_defect
    final_fail = ~df.final_pass
    n = int(truth.sum())
    res = []
    z_r = residual_z(df, "ridge")
    z_h = residual_z(df, "hgb")
    for name, s in [("residual ridge: max |z| over 4 readings", z_r.max(axis=1)),
                    ("residual ridge: bandgap only", z_r["bg_untrim_V"]),
                    ("residual hgb: max |z| over 4 readings", z_h.max(axis=1)),
                    ("rule: bandgap code at DAC edge (0 or 63)", df.bg_trim_code_opt.isin([0, 63]).astype(float))]:
        res.append(dict(method=name, **metrics(s.to_numpy(), truth, final_fail)))
    # unsupervised multivariate baseline on raw sort features
    X = df[MEASURES + FEATS + [t["code"] for t in TRIMS.values()]]
    iso = IsolationForest(n_estimators=300, random_state=0).fit(StandardScaler().fit_transform(X))
    res.append(dict(method="IsolationForest on raw features", **metrics(-iso.score_samples(StandardScaler().fit_transform(X)),
                                                                          truth, final_fail)))
    best = z_r.max(axis=1)
    out = [f"# Latent-defect screen ({len(df)} dies, {n} latent defects = {n / len(df):.2%})\n",
           "Scores are computed from sort-time data only; `latent_defect` is used to score.\n",
           "Metric note: recall_at_k flags the k highest-scoring dies where k = number of true latent defects.\n",
           pd.DataFrame(res).round(3).to_markdown(index=False),
           "\n## Operating points for the ridge residual screen (max |z|)\n",
           threshold_table(best, truth, final_fail).round(3).to_markdown(index=False),
           f"\nFinal-test fail rate overall: {final_fail.mean():.2%}; among true latent defects: {final_fail[truth].mean():.0%}"]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
