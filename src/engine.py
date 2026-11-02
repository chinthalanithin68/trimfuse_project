"""Backend for the UI: runs every model leave-wafer-out and returns one row per die.

Leave-wafer-out matters for the demo: the code shown for a die comes from a model that never saw that die's wafer.
MCU-B / MCU-C dies use the MCU-A trim models untuned; src/transfer.py measures how well that works (B: no tuning needed, C: ~10 dies help).
"""
import numpy as np
import pandas as pd

from anomaly import residual_z
from data import TRIMS, load_mcu, load_schema, wafer_folds
from drift import FEATS as DRIFT_FEATS, PARAMS, add_shift, shift_model
from fuse_check import SORT_FIELDS, validate_word
from trim_models import BEST_KIND, feature_sets, flow_iterations, make_model, predict_codes

BEST_TRIM_MODEL = BEST_KIND  # chosen from reports/02_trim_results.md
ANOMALY_Z_HOLD = 6.0  # screen flags exactly the 30 latent defects at z=4..8; 6 sits mid-plateau


def trim_predictions(df):
    """Predicted trim codes. MCU-A: wafer-grouped out-of-fold. Others: model fit on all of MCU-A."""
    out = {}
    a_mask = (df.derivative == "MCU-A").to_numpy()
    a = df[a_mask]
    for name, t in TRIMS.items():
        feats = feature_sets(name)["base"]  # "chained" (adds bg_untrim_V) measured no better
        kind = BEST_TRIM_MODEL[name]
        pred = np.zeros(len(df), dtype=int)
        a_pred = np.zeros(len(a), dtype=int)
        for tr, te in wafer_folds(a):
            m = make_model(kind).fit(a.iloc[tr][feats], a.iloc[tr][t["code"]])
            a_pred[te] = predict_codes(m, a.iloc[te][feats], t["bits"])
        pred[a_mask] = a_pred
        if (~a_mask).any():
            m = make_model(kind).fit(a[feats], a[t["code"]])
            pred[~a_mask] = predict_codes(m, df[~a_mask][feats], t["bits"])
        out[name] = pred
    return out


def drift_predictions(df):
    """Predicted sort->final shift per trim, leave-wafer-out. MCU-A: ridge on other A wafers.
    MCU-B/C: MCU-A ridge, linearly recalibrated on the derivative's dies from other wafers."""
    d = add_shift(df)
    a = d[d.derivative == "MCU-A"]
    res = {k: np.zeros(len(d)) for k in PARAMS}
    for k in PARAMS:
        for deriv, g in d.groupby("derivative"):
            for w in g.wafer_id.unique():
                te = g[g.wafer_id == w]
                if deriv == "MCU-A":
                    m = shift_model("ridge").fit(a[a.wafer_id != w][DRIFT_FEATS], a[a.wafer_id != w][f"{k}_shift"])
                    p = m.predict(te[DRIFT_FEATS])
                else:
                    m = shift_model("ridge").fit(a[DRIFT_FEATS], a[f"{k}_shift"])
                    other = g[g.wafer_id != w]
                    from sklearn.linear_model import LinearRegression
                    cal = LinearRegression().fit(m.predict(other[DRIFT_FEATS]).reshape(-1, 1), other[f"{k}_shift"])
                    p = cal.predict(m.predict(te[DRIFT_FEATS]).reshape(-1, 1))
                res[k][d.index.get_indexer(te.index)] = p
    return res


def decide(fuse_violations, z, code_err):
    """Tester decision for one die. code_err = |predicted - optimal| as revealed by the verification measurement."""
    if fuse_violations:
        return "BLOCK - fuse word invalid, do not program"
    if z >= ANOMALY_Z_HOLD:
        return "HOLD - latent-defect suspect, send to review"
    if code_err == 0:
        return "PASS - predicted code verified first time"
    if code_err <= 1:
        return "PASS - one local step after verify"
    return "FALLBACK - full binary search"


def build():
    """Return (per-die DataFrame, schema). Takes a few seconds; the app caches it."""
    df = load_mcu().reset_index(drop=True)
    schema = load_schema()
    pred = trim_predictions(df)
    drift = drift_predictions(df)
    z = residual_z(df, "ridge")
    out = df.copy()
    ai_ms = np.zeros(len(df))
    for name, t in TRIMS.items():
        out[f"{name}_pred"] = pred[name]
        out[f"{name}_err"] = np.abs(pred[name] - df[t["code"]].to_numpy())
        it = flow_iterations(pred[name], df[t["code"]].to_numpy(), t["bits"])
        out[f"{name}_ai_iters"] = it
        ai_ms += it * t["ms"]
    out["ai_trim_time_ms"] = ai_ms
    for k in PARAMS:
        out[f"{k}_drift_pred"] = drift[k]
    out["anomaly_z"] = z.max(axis=1)
    viol = []
    for r in out.itertuples():
        sc = {f: int(getattr(r, c)) for f, c in SORT_FIELDS.items()}
        viol.append(validate_word(r.fuse_map_hex, r.derivative, schema, sc))
    out["fuse_violations"] = viol
    out["worst_code_err"] = out[[f"{n}_err" for n in TRIMS]].max(axis=1)
    out["decision"] = [decide(v, zz, e) for v, zz, e in zip(out.fuse_violations, out.anomaly_z, out.worst_code_err)]
    return out, schema
