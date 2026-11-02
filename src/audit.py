"""Data-quality audit for the MCU trim & fuse dataset. Writes reports/01_data_quality_report.md."""
from pathlib import Path

import numpy as np
import pandas as pd

from data import clean_mcu, load_schema

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "reports" / "01_data_quality_report.md"

df = pd.read_csv(DATA / "mcu_trim_fuse.csv")
sch = load_schema()
L = []


def sec(t):
    L.append(f"\n## {t}\n")


def add(s=""):
    L.append(s)


add("# Data-Quality Audit - MCU trim & fuse dataset (mcu_trim_fuse.csv)")

sec("1. Shape and schema")
add(f"- mcu_trim_fuse: {df.shape}; fuse layout (embedded): {sch.shape}")
add(f"- dtypes: {df.dtypes.astype(str).value_counts().to_dict()}")

sec("2. Missing values")
n = df.isna().sum()
add(f"- columns with nulls: {n[n > 0].to_dict() or 'none'}")
add(f"- blank strings in object columns: {int(sum((df[c].astype(str).str.strip() == '').sum() for c in df.select_dtypes(exclude='number')))}")

sec("3. Duplicates and keys")
add(f"- duplicate rows: {df.duplicated().sum()}; duplicate die_id: {df.die_id.duplicated().sum()}")
add(f"- duplicate (wafer_id, die_x, die_y): {df.duplicated(['wafer_id','die_x','die_y']).sum()}")

sec("4. Structure")
g = df.groupby("derivative").agg(dies=("die_id", "size"), wafers=("wafer_id", "nunique"), pkg=("package", lambda s: sorted(s.unique())), grade=("temp_grade", lambda s: sorted(s.unique())))
add("```\n" + g.to_string() + "\n```")
add(f"- lots: {df.lot_id.nunique()}, wafers: {df.wafer_id.nunique()}, wafers spanning >1 derivative: {int((df.groupby('wafer_id').derivative.nunique() > 1).sum())}")
add(f"- derivative-to-package mapping unique: {df.groupby('derivative').package.nunique().to_dict()}")
r = np.hypot(df.x_mm, df.y_mm)
add(f"- radius_mm vs sqrt(x^2+y^2): max abs diff {np.abs(r - df.radius_mm).max():.3f} mm")
add(f"- ts_cal2_measured_on_tester True only for: {df.loc[df.ts_cal2_measured_on_tester, 'derivative'].unique().tolist()} "
    f"(share of that derivative: {df.loc[df.derivative=='MCU-C','ts_cal2_measured_on_tester'].mean():.2f})")

sec("5. Range checks (trim codes vs DAC bits, targets)")
for col, bits in [("bg_trim_code_opt", 6), ("hsi_trim_code_opt", 7), ("ldo_trim_code_opt", 5)]:
    s = df[col]
    add(f"- {col}: min={s.min()}, max={s.max()}, valid range 0..{2**bits-1}, out of range={int(((s<0)|(s>2**bits-1)).sum())}, "
        f"at range edge (0 or max)={int(((s==0)|(s==2**bits-1)).sum())}")
for col, tgt, tol in [("bg_trimmed_sort_V", 1.2, .005), ("hsi_trimmed_sort_MHz", 16, .16), ("ldo_trimmed_sort_V", 1.1, .022)]:
    err = (df[col] - tgt).abs()
    add(f"- {col}: mean abs error vs target {tgt} = {err.mean():.5f}, max={err.max():.5f}, dies beyond +/-{tol}: {int((err > tol).sum())}")
add(f"- sram_fail_bits range {df.sram_fail_bits.min()}..{df.sram_fail_bits.max()}; conv_trim_time_ms {df.conv_trim_time_ms.min()}..{df.conv_trim_time_ms.max()}")
add(f"- conv iterations (bg/hsi/ldo) range: {df.bg_conv_iterations.min()}-{df.bg_conv_iterations.max()}, {df.hsi_conv_iterations.min()}-{df.hsi_conv_iterations.max()}, {df.ldo_conv_iterations.min()}-{df.ldo_conv_iterations.max()}")

sec("6. Fuse word cross-check (decode fuse_map_hex and compare to columns)")
fw = pd.Series([int(h, 16) for h in df.fuse_map_hex], dtype=object, index=df.index)
for _, row in sch[sch.field.isin(["BG_TRIM", "HSI_TRIM", "LDO_TRIM", "TS_CAL1"])].iterrows():
    val = pd.Series([(w >> int(row.lsb)) & ((1 << int(row.nbits)) - 1) for w in fw], index=df.index)
    ref = {"BG_TRIM": "bg_trim_code_opt", "HSI_TRIM": "hsi_trim_code_opt", "LDO_TRIM": "ldo_trim_code_opt", "TS_CAL1": "ts_cal1_30C_counts"}[row.field]
    add(f"- {row.field} decoded vs {ref}: mismatches = {int((val != df[ref]).sum())}")
resv = pd.Series([(w >> 43) & ((1 << 21) - 1) for w in fw], index=df.index)
add(f"- RESERVED bits non-zero in main table: {int((resv != 0).sum())}; DEBUG_LOCK=0: {sum(((w >> 41) & 1) == 0 for w in fw)}")

sec("7. Label consistency")
add(f"- sort_pass=False: {int((~df.sort_pass).sum())}; final_pass=False: {int((~df.final_pass).sum())}; final fail though sort pass: {int((df.sort_pass & ~df.final_pass).sum())}")
bg_ok = (df.bg_final_V - 1.2).abs() <= .005
hsi_ok = (df.hsi_final_MHz / 16 - 1).abs() <= .01
ldo_ok = (df.ldo_final_V / 1.1 - 1).abs() <= .02
calc = bg_ok & hsi_ok & ldo_ok
add(f"- final_pass recomputed from spec vs stored: mismatches={int((calc != df.final_pass).sum())} (among sort-pass dies: {int(((calc != df.final_pass) & df.sort_pass).sum())})")
add(f"- final fail rate by derivative: {(1 - df.groupby('derivative').final_pass.mean()).round(4).to_dict()}")
add(f"- latent defects: {int(df.latent_defect.sum())} ({df.latent_defect.mean():.2%}); final fail among them: {(1 - df.loc[df.latent_defect,'final_pass'].mean()):.2f} vs {(1 - df.loc[~df.latent_defect,'final_pass'].mean()):.3f} for others")

sec("8. Outliers (IQR x 3) on features")
for c in ["pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel", "bg_untrim_V", "hsi_untrim_MHz", "ldo_untrim_V", "ts_cal1_30C_counts"]:
    q1, q3 = df[c].quantile([.25, .75])
    k = int(((df[c] < q1 - 3 * (q3 - q1)) | (df[c] > q3 + 3 * (q3 - q1))).sum())
    add(f"- {c}: extreme outliers = {k}")
add(f"- latent defects vs trim-code edge rate: {((df.bg_trim_code_opt.isin([0,63])) & df.latent_defect).sum()} defective dies with BG code at edge")

sec("9. Pattern checks (what the models should exploit)")
add(f"- corr(bg_untrim_V, bg_trim_code_opt) = {df.bg_untrim_V.corr(df.bg_trim_code_opt):.3f}")
add(f"- corr(hsi_untrim_MHz, hsi_trim_code_opt) = {df.hsi_untrim_MHz.corr(df.hsi_trim_code_opt):.3f}")
add(f"- corr(ldo_untrim_V, bg_untrim_V) = {df.ldo_untrim_V.corr(df.bg_untrim_V):.3f} (coupling)")
add(f"- corr(ts_cal1, ts_cal2) = {df.ts_cal1_30C_counts.corr(df.ts_cal2_110C_counts):.3f}")
d = (df.bg_final_V - df.bg_trimmed_sort_V)
add(f"- bandgap sort-to-final shift (mV) by derivative: {(d.groupby(df.derivative).mean()*1000).round(2).to_dict()}; corr(shift, radius) = {d.corr(df.radius_mm):.3f}")

sec("10. Cleaning step applied (data.clean_mcu, run on the raw file)")
_, clog = clean_mcu(df)
add("| action | count |")
add("|---|---|")
for k, v in clog.items():
    add(f"| {k} | {v} |")
add("")
add("Policy: duplicates are dropped (first kept); a row with a missing required value is dropped, never imputed, because a guessed "
    "trim code or fuse bit is unsafe; out-of-range DAC codes are dropped; text is stripped and fuse words normalised to 0x + 16 upper-case hex "
    "digits; outliers and fuse-vs-column mismatches are only counted. On this file every drop count is 0, so the cleaned data equals the raw data; "
    "tests/test_data_and_trim.py proves each action on deliberately dirty data.")

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
