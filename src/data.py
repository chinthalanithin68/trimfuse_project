"""Loading, validation, fuse-word decoding and wafer-grouped splits for the MCU trim & fuse data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PCM = ["pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel"]
POS = ["x_mm", "y_mm", "radius_mm"]

# Columns the dataset dictionary marks role=feature for this table (available at test time). Every model input must come from
# here (plus values derived from them); the label-role columns (trim codes, trimmed/final values, flags) are targets only.
FEATURE_COLUMNS = {"die_x", "die_y", "x_mm", "y_mm", "radius_mm", *PCM, "bg_untrim_V", "hsi_untrim_MHz", "ldo_untrim_V",
                   "ts_cal1_30C_counts", "sram_fail_bits"}
DERIVED_FEATURES = {"r2", "is_new", "is_c"}  # radius squared and derivative flags built inside the modules

# Per trim: untrimmed measurement, label column, DAC bits, target, tester cost per iteration (ms).
# The per-iteration costs reproduce conv_trim_time_ms = 1.2*bg + 2.0*hsi + 1.0*ldo iterations.
TRIMS = {
    "bg": dict(untrim="bg_untrim_V", code="bg_trim_code_opt", bits=6, ms=1.2, iters="bg_conv_iterations"),
    "hsi": dict(untrim="hsi_untrim_MHz", code="hsi_trim_code_opt", bits=7, ms=2.0, iters="hsi_conv_iterations"),
    "ldo": dict(untrim="ldo_untrim_V", code="ldo_trim_code_opt", bits=5, ms=1.0, iters="ldo_conv_iterations"),
}

REQUIRED = (["die_id", "wafer_id", "derivative", "latent_defect", "sort_pass", "final_pass", "fuse_map_hex",
             "conv_trim_time_ms"] + PCM + POS
            + [v for t in TRIMS.values() for v in (t["untrim"], t["code"], t["iters"])])


BOOL_COLUMNS = ["latent_defect", "sort_pass", "final_pass", "sram_repairable", "ts_cal2_measured_on_tester"]


def load_mcu(path=None, validate=True, clean=True):
    """Read mcu_trim_fuse.csv, apply the cleaning step, then validate. Raises ValueError on schema or range problems."""
    df = pd.read_csv(path or DATA_DIR / "mcu_trim_fuse.csv")
    if clean:
        df, _ = clean_mcu(df)
    if validate:
        validate_mcu(df)
    return df


def clean_mcu(raw):
    """Cleaning step. Returns (clean_df, log) where log counts every action taken (all zero on the shipped file).

    1. strip whitespace in text columns; normalise fuse words to '0x' + 16 upper-case hex digits
    2. parse boolean columns written as text; coerce numeric columns (bad text becomes NaN)
    3. drop exact duplicate rows, then rows repeating a die_id (first one kept)
    4. drop rows with a missing value in any REQUIRED column (no imputation: a guessed trim code or fuse bit is unsafe);
       missing values in other columns are kept and counted
    5. drop rows whose optimal trim code is outside its DAC range
    Not removed, only counted: statistical outliers (they are defect signal) and fuse words that disagree with the
    trim columns.
    """
    df = raw.copy()
    log = {"rows_in": len(df)}

    text = {"die_id", "lot_id", "wafer_id", "derivative", "package", "temp_grade", "fuse_map_hex"}
    for c in text & set(df.columns):  # only genuine text columns: .str.strip() would blank out numbers stored as object
        df[c] = df[c].astype("string").str.strip().astype(df[c].dtype)
    if "fuse_map_hex" in df:
        ok = df["fuse_map_hex"].notna()
        df.loc[ok, "fuse_map_hex"] = [f"0x{int(h, 16):016X}" for h in df.loc[ok, "fuse_map_hex"]]
    for c in BOOL_COLUMNS:
        if c in df and df[c].dtype != bool:
            df[c] = df[c].astype("string").str.lower().map({"true": True, "false": False, "1": True, "0": False}).astype("boolean")
    for c in df.columns:
        if c not in text | set(BOOL_COLUMNS) and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    log["duplicate_rows_dropped"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log["duplicate_die_id_dropped"] = int(df["die_id"].duplicated().sum()) if "die_id" in df else 0
    df = df.drop_duplicates(subset="die_id") if "die_id" in df else df

    req = [c for c in REQUIRED if c in df.columns]
    null_req = df[req].isna().any(axis=1)
    log["rows_dropped_missing_required"] = int(null_req.sum())
    log["missing_required_by_column"] = {k: int(v) for k, v in df[req].isna().sum().items() if v}
    df = df[~null_req]
    log["missing_in_other_columns_kept"] = {k: int(v) for k, v in df.isna().sum().items() if v}

    bad_code = pd.Series(False, index=df.index)
    for t in TRIMS.values():
        if t["code"] in df:
            bad_code |= (df[t["code"]] < 0) | (df[t["code"]] > 2 ** t["bits"] - 1)
    log["rows_dropped_code_out_of_range"] = int(bad_code.sum())
    df = df[~bad_code].reset_index(drop=True)
    for t in TRIMS.values():
        if t["code"] in df:
            df[t["code"]] = df[t["code"]].astype(int)

    outliers = 0
    for c in ["bg_untrim_V", "hsi_untrim_MHz", "ldo_untrim_V", "ts_cal1_30C_counts", *PCM]:
        if c in df:
            q1, q3 = df[c].quantile([.25, .75])
            outliers += int(((df[c] < q1 - 3 * (q3 - q1)) | (df[c] > q3 + 3 * (q3 - q1))).sum())
    log["extreme_outlier_values_kept"] = outliers
    words = [int(h, 16) for h in df["fuse_map_hex"]]
    mism = np.zeros(len(df), bool)
    for field, col in (("BG_TRIM", "bg_trim_code_opt"), ("HSI_TRIM", "hsi_trim_code_opt"), ("LDO_TRIM", "ldo_trim_code_opt"),
                       ("TS_CAL1", "ts_cal1_30C_counts")):
        r = next(x for x in _FUSE_LAYOUT if x[0] == field)
        mism |= np.array([(w >> r[1]) & ((1 << r[2]) - 1) for w in words]) != df[col].to_numpy()
    mism = int(mism.sum())
    log["fuse_word_vs_trim_column_mismatches_kept"] = int(mism)
    log["rows_out"] = len(df)
    return df, log


def validate_mcu(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["die_id"].duplicated().any():
        raise ValueError("duplicate die_id")
    nulls = df[REQUIRED].isna().sum()
    if nulls.any():
        raise ValueError(f"nulls in required columns: {nulls[nulls > 0].to_dict()}")
    for name, t in TRIMS.items():
        c = df[t["code"]]
        if (c < 0).any() or (c > 2 ** t["bits"] - 1).any():
            raise ValueError(f"{name} trim code outside 0..{2 ** t['bits'] - 1}")
    return df


def decode_fuse(hex_word, schema):
    """Split a 64-bit fuse word into {field: value} using the schema's lsb / nbits columns."""
    w = int(hex_word, 16)
    return {r.field: (w >> int(r.lsb)) & ((1 << int(r.nbits)) - 1) for r in schema.itertuples()}


# 64-bit fuse word layout. fuse_map_hex lives in mcu_trim_fuse.csv; the layout is documented in the dataset's
# data dictionary (field, lsb, nbits, description). Fixed fields differ per derivative; "per-die" ones come from sort.
_FUSE_LAYOUT = [
    ("BG_TRIM", 0, 6, "Bandgap trim code", "per-die", "per-die", "per-die"),
    ("HSI_TRIM", 6, 7, "HSI oscillator trim code", "per-die", "per-die", "per-die"),
    ("LDO_TRIM", 13, 5, "LDO trim code", "per-die", "per-die", "per-die"),
    ("TS_CAL1", 18, 12, "Temperature sensor reading at 30 C", "per-die", "per-die", "per-die"),
    ("FLASH_SIZE", 30, 2, "0=256 KB, 1=512 KB, 2=1 MB", "1", "0", "2"),
    ("PERIPH_EN", 32, 8, "Peripheral enable mask", "0xFF", "0x3F", "0xFF"),
    ("TEMP_GRADE", 40, 1, "1 = automotive grade", "0", "0", "1"),
    ("DEBUG_LOCK", 41, 1, "1 = debug port locked (required in production)", "1", "1", "1"),
    ("RDP", 42, 1, "1 = read-out protection enabled", "0", "0", "1"),
    ("RESERVED", 43, 21, "Reserved, must be 0", "0", "0", "0"),
]


def load_schema():
    return pd.DataFrame(_FUSE_LAYOUT, columns=["field", "lsb", "nbits", "description", "MCU-A", "MCU-B", "MCU-C"])


def wafer_folds(df, n_splits=5):
    """Yield (train_idx, test_idx) positional indices; no wafer is in both sets."""
    n = min(n_splits, df["wafer_id"].nunique())
    yield from GroupKFold(n_splits=n).split(df, groups=df["wafer_id"])


def wafer_holdout(df, test_frac=0.25, seed=0):
    """Single wafer-grouped holdout. Returns (train_df, test_df)."""
    rng = np.random.default_rng(seed)
    wafers = df["wafer_id"].unique()
    test_w = set(rng.choice(wafers, max(1, int(len(wafers) * test_frac)), replace=False))
    m = df["wafer_id"].isin(test_w)
    return df[~m], df[m]
