"""Streamlit UI: run with  streamlit run app.py"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import engine  # noqa: E402
from data import TRIMS, decode_fuse  # noqa: E402
from drift import PARAMS  # noqa: E402
from fuse_check import SORT_FIELDS, expected_fixed_fields, validate_word  # noqa: E402
from fuse_gen import generate_fuse_word  # noqa: E402

st.set_page_config(page_title="AI Trim & Fuse - MCU", layout="wide")


@st.cache_data(show_spinner="Training models leave-wafer-out (about 30 s, once)...")
def load():
    return engine.build()


df, schema = load()
RULE_TEXT = {"FLASH_SIZE": "flash size does not match derivative", "PERIPH_EN": "peripheral enable mask wrong",
             "TEMP_GRADE": "temperature grade bit wrong", "DEBUG_LOCK": "debug lock not set (required in production)",
             "RDP": "read-out protection bit wrong for derivative", "RESERVED": "reserved bits must be 0",
             "SORT_MISMATCH": "trim field differs from the code found at sort"}

st.title("AI-Optimized Trim & Fuse - MCU")
st.caption("Synthetic data. Every prediction shown for a die comes from a model that never saw that die's wafer. "
           "Fuses are one-time programmable: the AI never blows a fuse without a verify step and the rule check.")
tab_over, tab_die, tab_fuse, tab_rep = st.tabs(["Overview", "Die explorer", "Fuse checker", "Method & reports"])

# ------------------------------------------------------------------ Overview
with tab_over:
    conv, ai = df.conv_trim_time_ms.mean(), df.ai_trim_time_ms.mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dies", f"{len(df):,}")
    c2.metric("Trim time / die (conventional)", f"{conv:.1f} ms")
    c3.metric("Trim time / die (predictive)", f"{ai:.1f} ms", f"-{1 - ai / conv:.0%}", delta_color="inverse")
    c4.metric("Held for review (latent suspects)", int(df.decision.str.startswith("HOLD").sum()))

    rows = []
    for n, t in TRIMS.items():
        rows.append({"trim": n, "model": engine.BEST_TRIM_MODEL[n], "exact %": (df[f"{n}_err"] == 0).mean() * 100,
                     "within +/-1 code %": (df[f"{n}_err"] <= 1).mean() * 100,
                     "iterations conv": df[t["iters"]].mean(), "iterations AI": df[f"{n}_ai_iters"].mean(),
                     "ms conv": (df[t["iters"]] * t["ms"]).mean(), "ms AI": (df[f"{n}_ai_iters"] * t["ms"]).mean()})
    st.subheader("Predictive trim, all derivatives")
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
    by = df.groupby("derivative").agg(dies=("die_id", "size"), conv_ms=("conv_trim_time_ms", "mean"),
                                      ai_ms=("ai_trim_time_ms", "mean")).round(2)
    st.subheader("By derivative")
    st.dataframe(by, use_container_width=True)
    st.caption("MCU-A is scored out-of-fold across its 15 wafers. MCU-B and MCU-C use the MCU-A trim models with no tuning, "
               "validated in the 'Trim transfer to MCU-B/C' report: MCU-B needs no tuning; MCU-C is about 9.3 ms untuned and "
               "about 8.4 ms with 10 tuning dies.")
    st.subheader("Tester decisions")
    dc = df.decision.value_counts().rename_axis("decision").reset_index(name="dies")
    st.altair_chart(alt.Chart(dc).mark_bar().encode(x="dies:Q", y=alt.Y("decision:N", sort="-x", title=None),
                                                    tooltip=["decision", "dies"]), use_container_width=True)

# ------------------------------------------------------------------ Die explorer
with tab_die:
    left, right = st.columns([1, 2])
    with left:
        deriv = st.selectbox("Derivative", sorted(df.derivative.unique()), key="explorer_derivative")
        wafers = sorted(df[df.derivative == deriv].wafer_id.unique())
        wafer = st.selectbox("Wafer", wafers)
        w = df[df.wafer_id == wafer]
        only_flag = st.checkbox("Only dies that are not a clean PASS", value=False)
        if only_flag:
            w = w[~w.decision.str.startswith("PASS - predicted")]
        die = st.selectbox("Die", w.die_id.tolist())
    d = df[df.die_id == die].iloc[0]
    with right:
        st.subheader(f"{die}  |  {d.derivative}  |  {d.package}  |  wafer {d.wafer_id}")
        dec = d.decision
        (st.error if dec.startswith(("BLOCK", "HOLD")) else st.warning if dec.startswith("FALLBACK") else st.success)(dec)

    a, b = st.columns(2)
    with a:
        st.markdown("**Predictive trim**")
        t_rows = []
        for n, t in TRIMS.items():
            t_rows.append({"trim": n, "untrimmed": round(d[t["untrim"]], 4), "predicted code": int(d[f"{n}_pred"]),
                           "optimal code": int(d[t["code"]]), "error": int(d[f"{n}_err"]),
                           "iterations (conv -> AI)": f"{int(d[t['iters']])} -> {int(d[f'{n}_ai_iters'])}"})
        st.dataframe(pd.DataFrame(t_rows), hide_index=True, use_container_width=True)
        st.caption(f"Trim time: {d.conv_trim_time_ms:.1f} ms conventional vs {d.ai_trim_time_ms:.1f} ms predictive. "
                   "Flow: apply predicted code and verify; off by 1 = one local step; further off = full search.")
        st.markdown("**Drift-aware targets** (package shift predicted at sort)")
        dr = []
        for k, p in PARAMS.items():
            pred, actual = d[f"{k}_drift_pred"], (d[p["fin"]] - d[p["sort"]]) * p["scale"]
            dr.append({"trim": k, "predicted shift": round(pred, 3), "actual shift": round(actual, 3), "unit": p["unit"]})
        st.dataframe(pd.DataFrame(dr), hide_index=True, use_container_width=True)
    with b:
        st.markdown("**Latent-defect screen**")
        st.metric("Anomaly score (robust z)", f"{d.anomaly_z:.1f}", "flagged" if d.anomaly_z >= engine.ANOMALY_Z_HOLD else "normal",
                  delta_color="inverse" if d.anomaly_z >= engine.ANOMALY_Z_HOLD else "off")
        st.caption(f"Hold threshold z >= {engine.ANOMALY_Z_HOLD:g}. Ground truth (scoring only): latent_defect = {bool(d.latent_defect)}, "
                   f"final test {'pass' if d.final_pass else 'FAIL'}.")
        st.markdown("**Fuse word**")
        viol = d.fuse_violations
        st.code(d.fuse_map_hex)
        (st.error if viol else st.success)("Violations: " + ", ".join(viol) if viol else "All rules satisfied")
        f = decode_fuse(d.fuse_map_hex, schema)
        st.dataframe(pd.DataFrame({"field": f.keys(), "value": [hex(v) for v in f.values()]}), hide_index=True,
                     use_container_width=True, height=250)

    st.markdown("**Wafer map**")
    metric = st.radio("Colour by", ["anomaly_z", "worst_code_err", "ai_trim_time_ms"], horizontal=True)
    wm = df[df.wafer_id == d.wafer_id]
    chart = alt.Chart(wm).mark_square(size=90).encode(
        x=alt.X("die_x:O", title="die_x"), y=alt.Y("die_y:O", sort="descending", title="die_y"),
        color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="viridis")),
        stroke=alt.condition(alt.datum.die_id == die, alt.value("red"), alt.value(None)),
        strokeWidth=alt.value(2),
        tooltip=["die_id", "decision", "anomaly_z", "worst_code_err"]).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

# ------------------------------------------------------------------ Fuse checker
with tab_fuse:
    st.subheader("Validate a fuse word before it is programmed")
    c1, c2 = st.columns([1, 1])
    with c1:
        src = st.radio("Start from", ["A real die", "Type a word"], horizontal=True)
        if src == "A real die":
            die_f = st.selectbox("Die", df.die_id.tolist(), key="fuse_die")
            r = df[df.die_id == die_f].iloc[0]
            word, fd = r.fuse_map_hex, r.derivative
            sort = {k: int(r[c]) for k, c in SORT_FIELDS.items()}
        else:
            word = st.text_input("64-bit fuse word (hex)", "0x000002FF504A2927")
            fd = st.selectbox("Derivative", ["MCU-A", "MCU-B", "MCU-C"], key="fuse_derivative")
            use_sort = st.checkbox("Also check against sort-log codes")
            sort = None
            if use_sort:
                sort = {k: st.number_input(k, min_value=0, value=0, step=1) for k in SORT_FIELDS}
        flip = st.number_input("Flip one bit (-1 = none) to see which rule catches it", -1, 63, -1)
        try:
            w_int = int(word, 16)
            if flip >= 0:
                w_int ^= 1 << int(flip)
            word_t = f"0x{w_int:016X}"
        except ValueError:
            word_t = None
            st.error("Not a valid hex word")
    with c2:
        if word_t:
            v = validate_word(word_t, fd, schema, sort)
            st.code(word_t)
            if v:
                st.error("BLOCK - do not program")
                for x in v:
                    st.write(f"- **{x}**: {RULE_TEXT[x]}")
            else:
                st.success("APPROVED by the rule check (still requires the verify measurement)")
            if sort is None:
                st.info("No sort log supplied: per-die trim fields (bits 0-29) cannot be checked, "
                        "so a flipped trim bit passes the schema rules.")
            exp = expected_fixed_fields(schema, fd)
            fields = decode_fuse(word_t, schema)
            st.dataframe(pd.DataFrame({"field": list(fields), "value": [hex(x) for x in fields.values()],
                                       "expected": [hex(exp[k]) if k in exp else "per-die" for k in fields]}),
                         hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Generate a fuse word from codes")
    st.caption("The generator fills the fixed fields from the derivative spec and the per-die fields from the sort codes, "
               "then the validator checks the result.")
    g1, g2 = st.columns(2)
    with g1:
        gd = st.selectbox("Derivative", ["MCU-A", "MCU-B", "MCU-C"], key="gen_derivative")
        gcodes = {"BG_TRIM": st.number_input("BG_TRIM (6 bit, 0-63)", 0, 63, 32, key="gen_bg"),
                  "HSI_TRIM": st.number_input("HSI_TRIM (7 bit, 0-127)", 0, 127, 64, key="gen_hsi"),
                  "LDO_TRIM": st.number_input("LDO_TRIM (5 bit, 0-31)", 0, 31, 16, key="gen_ldo"),
                  "TS_CAL1": st.number_input("TS_CAL1 (12 bit, 0-4095)", 0, 4095, 1050, key="gen_ts")}
    with g2:
        gw = generate_fuse_word(gd, gcodes, schema)
        st.code(gw)
        gv = validate_word(gw, gd, schema, gcodes)
        (st.error if gv else st.success)("Violations: " + ", ".join(gv) if gv else "Generated word passes every rule")

# ------------------------------------------------------------------ Reports
with tab_rep:
    names = {"Data-quality audit": "01_data_quality_report.md", "Predictive trim": "02_trim_results.md",
             "Drift & transfer": "03_drift_results.md", "Temperature sensor": "05_ts_cal_results.md",
             "Trim transfer to MCU-B/C": "04_transfer_results.md", "Fuse validator": "06_fuse_check_results.md",
             "Fuse generator": "07_fuse_gen_results.md", "Anomaly screen": "08_anomaly_results.md",
             "Baseline comparison + Cpk": "09_baseline_comparison.md"}
    pick = st.selectbox("Report", list(names))
    p = ROOT / "reports" / names[pick]
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Not generated yet.")
    st.markdown("---\n**Models are offline**: scikit-learn ridge / gradient boosting run locally, no API calls at inference. "
                "Claude was used as the development assistant only.")
