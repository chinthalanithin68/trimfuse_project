"""Fuse-map generator: build the 64-bit fuse word from the derivative spec and a die's trim codes.

The fixed fields (flash size, peripheral mask, grade, debug lock, RDP, reserved) come from the derivative's column
of the fuse layout; the per-die fields (bandgap, HSI, LDO, temperature-sensor cal point) come from the sort log.
Correctness check: regenerate all 8,420 words in mcu_trim_fuse.csv and compare with the stored fuse_map_hex, then run
every generated word through the validator.

Run: python src/fuse_gen.py  -> writes reports/07_fuse_gen_results.md
"""
from pathlib import Path

import pandas as pd

from data import load_mcu, load_schema
from fuse_check import SORT_FIELDS, expected_fixed_fields, validate_word

REPORT = Path(__file__).resolve().parents[1] / "reports" / "07_fuse_gen_results.md"
PER_DIE = ("BG_TRIM", "HSI_TRIM", "LDO_TRIM", "TS_CAL1")


def generate_fuse_word(derivative, codes, schema):
    """codes: {'BG_TRIM': int, 'HSI_TRIM': int, 'LDO_TRIM': int, 'TS_CAL1': int}. Returns '0x%016X'.
    Raises ValueError for an unknown derivative, a missing/extra per-die field, or a value that does not fit its field."""
    if derivative not in schema.columns[4:]:
        raise ValueError(f"unknown derivative {derivative!r}")
    if set(codes) != set(PER_DIE):
        raise ValueError(f"codes must be exactly {PER_DIE}")
    fixed = expected_fixed_fields(schema, derivative)
    word = 0
    for r in schema.itertuples():
        value = codes[r.field] if r.field in codes else fixed[r.field]
        if not 0 <= int(value) < (1 << int(r.nbits)):
            raise ValueError(f"{r.field}={value} does not fit in {r.nbits} bits")
        word |= int(value) << int(r.lsb)
    return f"0x{word:016X}"


def codes_from_row(row):
    return {k: int(row[c]) for k, c in SORT_FIELDS.items()}


def roundtrip(df, schema):
    gen = [generate_fuse_word(r.derivative, codes_from_row(r._asdict()), schema) for r in df.itertuples()]
    gen = pd.Series(gen, index=df.index)
    violations = [validate_word(w, d, schema, codes_from_row(r)) for w, d, (_, r) in zip(gen, df.derivative, df.iterrows())]
    return gen, violations


def main():
    df, schema = load_mcu(), load_schema()
    gen, viol = roundtrip(df, schema)
    same = (gen == df.fuse_map_hex.str.upper().str.replace("0X", "0x")).sum()
    ok = sum(1 for v in viol if not v)
    bad_inputs = {}
    base = codes_from_row(df.iloc[0])
    for label, args in {"bandgap code 64 (6-bit field)": ("MCU-A", {**base, "BG_TRIM": 64}),
                        "negative LDO code": ("MCU-A", {**base, "LDO_TRIM": -1}),
                        "unknown derivative": ("MCU-Z", base),
                        "missing HSI code": ("MCU-A", {k: v for k, v in base.items() if k != "HSI_TRIM"})}.items():
        try:
            generate_fuse_word(*args, schema)
            bad_inputs[label] = "ACCEPTED (bug)"
        except ValueError as e:
            bad_inputs[label] = f"rejected: {e}"
    out = ["# Fuse-map generator results\n",
           f"- Words regenerated from derivative spec + sort codes: {len(df)}",
           f"- Identical to the stored `fuse_map_hex`: {same}/{len(df)}",
           f"- Generated words that pass the validator (with the sort log): {ok}/{len(df)}",
           "\n## Invalid inputs\n"] + [f"- {k}: {v}" for k, v in bad_inputs.items()]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
