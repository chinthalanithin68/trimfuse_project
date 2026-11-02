import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly import metrics, threshold_table  # noqa: E402
from data import load_mcu, load_schema  # noqa: E402
from fuse_check import expected_fixed_fields, make_test_set, single_bit_mutation, validate_table, validate_word  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return load_mcu(), load_schema()


def good_word(df, schema, deriv="MCU-A"):
    r = df[df.derivative == deriv].iloc[0]
    sc = {"BG_TRIM": r.bg_trim_code_opt, "HSI_TRIM": r.hsi_trim_code_opt, "LDO_TRIM": r.ldo_trim_code_opt,
          "TS_CAL1": r.ts_cal1_30C_counts}
    return r, sc


def test_schema_expectations(ctx):
    _, schema = ctx
    c = expected_fixed_fields(schema, "MCU-C")
    assert c["RDP"] == 1 and c["TEMP_GRADE"] == 1 and c["FLASH_SIZE"] == 2
    assert "BG_TRIM" not in c  # per-die fields are not fixed


def test_good_word_approved(ctx):
    df, schema = ctx
    r, sc = good_word(df, schema)
    assert validate_word(r.fuse_map_hex, r.derivative, schema, sc) == []


@pytest.mark.parametrize("bit,rule", [(41, "DEBUG_LOCK"), (50, "RESERVED"), (30, "FLASH_SIZE"), (35, "PERIPH_EN")])
def test_single_bit_flip_names_the_rule(ctx, bit, rule):
    df, schema = ctx
    r, sc = good_word(df, schema)
    bad = f"0x{int(r.fuse_map_hex, 16) ^ (1 << bit):016X}"
    assert rule in validate_word(bad, r.derivative, schema, sc)


def test_trim_flip_needs_sort_log(ctx):
    df, schema = ctx
    r, sc = good_word(df, schema)
    bad = f"0x{int(r.fuse_map_hex, 16) ^ 1:016X}"
    assert validate_word(bad, r.derivative, schema) == []  # schema rules cannot see per-die values
    assert validate_word(bad, r.derivative, schema, sc) == ["SORT_MISMATCH"]


def test_rdp_required_for_automotive(ctx):
    df, schema = ctx
    r, sc = good_word(df, schema, "MCU-C")
    bad = f"0x{int(r.fuse_map_hex, 16) & ~(1 << 42):016X}"
    assert "RDP" in validate_word(bad, "MCU-C", schema, sc)


def test_wrong_derivative_is_rejected(ctx):
    df, schema = ctx
    r, sc = good_word(df, schema, "MCU-A")
    assert validate_word(r.fuse_map_hex, "MCU-B", schema, sc)  # an A word is not a valid B word


def test_injected_errors_all_caught_no_false_alarms(ctx):
    df, schema = ctx
    aud = make_test_set(df, n=600, seed=3)
    assert aud.has_error.sum() > 100
    flagged = validate_table(aud, df, schema).map(bool)
    assert (flagged == aud.has_error).all()


def test_every_bit_position_detected_with_sort_log(ctx):
    df, schema = ctx
    assert (single_bit_mutation(df, schema, n=20) == 1.0).all()


def test_anomaly_metrics_perfect_and_threshold_table():
    truth = pd.Series([True, True, False, False, False])
    ff = pd.Series([True, False, False, False, False])
    score = np.array([9.0, 8.0, 1.0, 0.5, 0.2])
    m = metrics(score, truth, ff)
    assert m["auc"] == 1.0 and m["recall_at_k"] == 1.0 and m["final_fail_in_top_k"] == 0.5
    t = threshold_table(pd.Series(score), truth, ff, thresholds=(5,))
    assert t.loc[0, "caught"] == 2 and t.loc[0, "precision"] == 1.0
