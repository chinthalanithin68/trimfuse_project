"""Rule-based fuse-map validator.

Fuses are one-time programmable, so this gate is deterministic and explainable (no ML): every rule is read from
the fuse layout (data.load_schema) or the sort log (mcu_trim_fuse.csv). A word is approved only if no rule is violated.

Run: python src/fuse_check.py  -> writes reports/06_fuse_check_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import decode_fuse, load_mcu, load_schema

REPORT = Path(__file__).resolve().parents[1] / "reports" / "06_fuse_check_results.md"

# fuse field -> column in the sort log that holds the code that must have been programmed
SORT_FIELDS = {"BG_TRIM": "bg_trim_code_opt", "HSI_TRIM": "hsi_trim_code_opt", "LDO_TRIM": "ldo_trim_code_opt",
               "TS_CAL1": "ts_cal1_30C_counts"}
# injected error type -> rule expected to explain it
RULE_FOR_ERROR = {"wrong_flash_size": "FLASH_SIZE", "missing_debug_lock": "DEBUG_LOCK", "periph_mask_mismatch": "PERIPH_EN",
                  "reserved_bit_set": "RESERVED", "rdp_missing_automotive": "RDP", "trim_field_corrupt": "SORT_MISMATCH"}


def expected_fixed_fields(schema, derivative):
    """Fields fixed per derivative in the schema (everything not marked per-die)."""
    out = {}
    for field, v in zip(schema["field"], schema[derivative].astype(str).str.strip()):
        if v != "per-die":
            out[field] = int(v, 0)
    return out


def validate_word(hex_word, derivative, schema, sort_codes=None):
    """Return the list of violated rule ids (empty = approved).
    sort_codes: dict with the sort-log codes {'BG_TRIM': .., ...}; without it the trim fields cannot be checked."""
    f = decode_fuse(hex_word, schema)
    bad = [field for field, want in expected_fixed_fields(schema, derivative).items() if f[field] != want]
    if sort_codes is not None and any(f[k] != v for k, v in sort_codes.items()):
        bad.append("SORT_MISMATCH")
    return bad


def validate_table(words, main, schema, use_sort=True):
    """words: DataFrame(die_id, derivative, fuse_map_hex). Returns Series of lists of violated rule ids."""
    sort = main.set_index("die_id")
    out = []
    for r in words.itertuples():
        sc = None
        if use_sort:
            s = sort.loc[r.die_id]
            sc = {k: int(s[c]) for k, c in SORT_FIELDS.items()}
        out.append(validate_word(r.fuse_map_hex, r.derivative, schema, sc))
    return pd.Series(out, index=words.index)


def single_bit_mutation(main, schema, n=300, seed=0, use_sort=True):
    """Flip every one of 64 bits on n known-good words; report detection rate by schema field."""
    sample = main.sample(n, random_state=seed)
    rows = []
    for r in sample.itertuples():
        sc = {k: int(getattr(r, c)) for k, c in SORT_FIELDS.items()} if use_sort else None
        w = int(r.fuse_map_hex, 16)
        for bit in range(64):
            hit = bool(validate_word(f"0x{w ^ (1 << bit):016X}", r.derivative, schema, sc))
            field = next(x.field for x in schema.itertuples() if x.lsb <= bit < x.lsb + x.nbits)
            rows.append((field, hit))
    d = pd.DataFrame(rows, columns=["field", "detected"])
    return d.groupby("field").detected.mean()


ERROR_TYPES = list(RULE_FOR_ERROR)


def inject_error(word, etype, rng):
    """Corrupt a good 64-bit word the way a bad fuse-map generator or a bit upset would."""
    if etype == "wrong_flash_size":
        cur = (word >> 30) & 3
        return (word & ~(3 << 30)) | (((cur + 1) % 3) << 30)
    if etype == "missing_debug_lock":
        return word & ~(1 << 41)
    if etype == "periph_mask_mismatch":
        return word ^ (int(rng.integers(1, 255)) << 32)
    if etype == "trim_field_corrupt":
        return word ^ (1 << int(rng.integers(0, 18)))
    if etype == "reserved_bit_set":
        return word | (1 << int(rng.integers(43, 64)))
    if etype == "rdp_missing_automotive":
        return word & ~(1 << 42)
    raise ValueError(etype)


def make_test_set(main, n=1500, error_rate=0.25, seed=0):
    """Sample n real fuse words from mcu_trim_fuse.csv and corrupt a share of them.
    (rdp_missing_automotive only applies to MCU-C, which is the only derivative with RDP set.)"""
    rng = np.random.default_rng(seed)
    s = main.sample(n, random_state=seed)
    rows = []
    for r in s.itertuples():
        w, et = int(r.fuse_map_hex, 16), "none"
        if rng.random() < error_rate:
            et = str(rng.choice(ERROR_TYPES))
            if et == "rdp_missing_automotive" and r.derivative != "MCU-C":
                et = "missing_debug_lock"
            w = inject_error(w, et, rng)
        rows.append(dict(die_id=r.die_id, derivative=r.derivative, fuse_map_hex=f"0x{w:016X}", has_error=et != "none", error_type=et))
    return pd.DataFrame(rows)


def main():
    main_df, schema = load_mcu(), load_schema()
    aud = make_test_set(main_df)
    viol = validate_table(aud, main_df, schema)
    flagged = viol.map(bool)
    clean = ~aud.has_error
    out = ["# Fuse-map validator results\n",
           "Test set: 1,500 real fuse words from mcu_trim_fuse.csv, about 25% corrupted by the 6 error types in "
           "`inject_error` (seeded, regenerated on each run).\n",
           f"Words: {len(aud)}, injected errors: {int(aud.has_error.sum())} ({aud.has_error.mean():.1%}).\n",
           f"- Errors caught: {int((flagged & aud.has_error).sum())}/{int(aud.has_error.sum())}",
           f"- False alarms on clean words: {int((flagged & clean).sum())}/{int(clean.sum())}"]
    exact = np.mean([RULE_FOR_ERROR[e] in v for e, v in zip(aud.error_type[aud.has_error], viol[aud.has_error])])
    out.append(f"- Fired rule names the injected error type: {exact:.1%}")
    ct = pd.crosstab(aud.error_type[aud.has_error], viol[aud.has_error].map(lambda v: "+".join(v)))
    out += ["\nInjected error type vs rule(s) fired:\n", ct.to_markdown()]
    no_sort = validate_table(aud, main_df, schema, use_sort=False).map(bool)
    out += ["\n## Without the sort log (schema rules only)\n",
            f"- Errors caught: {int((no_sort & aud.has_error).sum())}/{int(aud.has_error.sum())}; "
            f"missed types: {aud.error_type[aud.has_error & ~no_sort].value_counts().to_dict()}",
            "\n## Single-bit-flip mutation test (any of 64 bits flipped on 300 good words)\n",
            "With sort log:\n", single_bit_mutation(main_df, schema).round(3).to_markdown(),
            "\nSchema rules only:\n", single_bit_mutation(main_df, schema, use_sort=False).round(3).to_markdown()]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
