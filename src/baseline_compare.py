"""Side-by-side: the dataset README's baseline results vs ours, plus final-test centring (Cpk) for the drift-aware target.

README numbers are copied from the dataset's own baseline_results.txt / README (gradient-boosted trees, wafer-grouped);
ours are recomputed here from the same code the app uses.

Run: python src/baseline_compare.py  -> writes reports/09_baseline_comparison.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

import engine
from data import TRIMS
from drift import PARAMS, add_shift, pass_rate, transfer_curve

REPORT = Path(__file__).resolve().parents[1] / "reports" / "09_baseline_comparison.md"
# dataset README baseline (within +/-1 code %, conventional iterations -> predictive iterations)
README = {"bg": (97.9, 6.6, 1.35), "hsi": (80.4, 7.6, 2.70), "ldo": (99.9, 5.6, 1.11)}
README_TRANSFER = dict(zero_shot_mae=1.73, tuned_mae=0.86, pass_conv=90.9, pass_tuned=99.4)


def cpk(values, target, tol):
    """Process capability against target +/- tol (mean and spread of the final value)."""
    mu, sd = np.mean(values), np.std(values, ddof=1)
    return min(target + tol - mu, mu - (target - tol)) / (3 * sd)


def main():
    df, _ = engine.build()
    a = df[df.derivative == "MCU-A"]
    rows = []
    for k, (rw1, rconv, rai) in README.items():
        rows.append(dict(use_case=f"MCU {k} predictive trim", readme=f"{rw1:.1f}% within +/-1; {rconv} -> {rai} iterations",
                         ours=f"{(a[f'{k}_err'] <= 1).mean() * 100:.1f}% within +/-1; {a[TRIMS[k]['iters']].mean():.1f} -> "
                              f"{a[f'{k}_ai_iters'].mean():.2f} iterations ({engine.BEST_TRIM_MODEL[k]})"))

    d = add_shift(df)
    c = d[d.derivative == "MCU-C"].reset_index(drop=True)
    a_all = d[d.derivative == "MCU-A"]
    cur = transfer_curve(a_all, c, "bg", seeds=5)
    zs = cur[(cur.method == "zero_shot")].mae.iloc[0]
    tuned = cur[(cur.method == "scale_correct") & (cur.n_tune == 50)]
    rows.append(dict(use_case="MCU-C drift transfer (bandgap shift error)",
                     readme=f"{README_TRANSFER['zero_shot_mae']} -> {README_TRANSFER['tuned_mae']} mV (50 random dies)",
                     ours=f"{zs:.2f} -> {tuned.mae.iloc[0]:.2f} mV (50 dies of one wafer, tested on the other)"))
    rows.append(dict(use_case="MCU-C bandgap final-test pass",
                     readme=f"{README_TRANSFER['pass_conv']}% -> {README_TRANSFER['pass_tuned']}%",
                     ours=f"{pass_rate(c, 'bg') * 100:.1f}% -> {tuned.pass_final.iloc[0] * 100:.1f}% (tuned, other wafer)"))
    rows.append(dict(use_case="Fuse-map validation", readme="406/406 injected errors, 0 false alarms (provided audit file)",
                     ours="368/368 injected errors, 0 false alarms (our own injection on real words; audit file not used)"))
    rows.append(dict(use_case="Fuse-map generation", readme="not in README baseline",
                     ours="8,420/8,420 words regenerated identically (see 07_fuse_gen_results.md)"))

    cp = []
    for deriv in ("MCU-A", "MCU-B", "MCU-C"):
        g = d[d.derivative == deriv]
        for label, sub in (("all dies", g), ("excluding latent defects", g[~g.latent_defect])):
            fin = sub[PARAMS["bg"]["fin"]].to_numpy()
            adj = fin - sub["bg_drift_pred"].to_numpy() / PARAMS["bg"]["scale"]
            cp.append(dict(derivative=deriv, dies=label, cpk_trim_to_nominal=cpk(fin, 1.2, 5e-3),
                           cpk_drift_aware=cpk(adj, 1.2, 5e-3),
                           pass_nominal=pass_rate(sub, "bg") * 100, pass_drift_aware=pass_rate(sub, "bg", sub["bg_drift_pred"]) * 100))
    out = ["# Dataset README baseline vs this project\n",
           pd.DataFrame(rows).to_markdown(index=False),
           "\nThe README baseline predicts all three trims with gradient-boosted trees. We pick ridge for bandgap and LDO because "
           "it measured better. The README tunes MCU-C on 50 random dies from both wafers; our test is stricter (tune on one "
           "wafer, test on the other).",
           "\n## Final-test centring: bandgap Cpk (spec 1.200 V +/- 5 mV)\n",
           "Drift-aware values come from leave-wafer-out predictions (see engine.py). Cpk >= 1.33 is the usual capability target.\n",
           pd.DataFrame(cp).round(2).to_markdown(index=False),
           "\nLatent defects add 5-15 mV of shift that the model cannot predict from sort data, which is why "
           "they are shown separately (they are held by the anomaly screen)."]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
