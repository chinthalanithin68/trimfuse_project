import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import TRIMS, decode_fuse, load_mcu, load_schema, validate_mcu, wafer_folds, wafer_holdout  # noqa: E402
from trim_models import feature_sets, flow_iterations, make_model, predict_codes  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return load_mcu()


def test_load_shape_and_keys(df):
    assert df.shape == (8420, 42)
    assert not df.die_id.duplicated().any()


def test_validate_rejects_bad_code(df):
    bad = df.copy()
    bad.loc[0, "bg_trim_code_opt"] = 64  # 6-bit DAC max is 63
    with pytest.raises(ValueError, match="bg trim code"):
        validate_mcu(bad)


def test_validate_rejects_missing_column(df):
    with pytest.raises(ValueError, match="missing columns"):
        validate_mcu(df.drop(columns=["bg_untrim_V"]))


def test_validate_rejects_nulls(df):
    bad = df.copy()
    bad.loc[3, "pcm_tox_A"] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        validate_mcu(bad)


def test_fuse_decode_matches_trim_columns(df):
    schema = load_schema()
    for row in df.sample(200, random_state=1).itertuples():
        f = decode_fuse(row.fuse_map_hex, schema)
        assert f["BG_TRIM"] == row.bg_trim_code_opt
        assert f["HSI_TRIM"] == row.hsi_trim_code_opt
        assert f["LDO_TRIM"] == row.ldo_trim_code_opt
        assert f["RESERVED"] == 0 and f["DEBUG_LOCK"] == 1


def test_wafer_folds_never_share_a_wafer(df):
    for tr, te in wafer_folds(df):
        assert set(df.wafer_id.iloc[tr]).isdisjoint(df.wafer_id.iloc[te])


def test_wafer_holdout_disjoint(df):
    tr, te = wafer_holdout(df)
    assert set(tr.wafer_id).isdisjoint(te.wafer_id) and len(tr) + len(te) == len(df)


def test_conv_time_formula_matches_per_iteration_cost(df):
    t = sum(df[v["iters"]] * v["ms"] for v in TRIMS.values()).round(1)
    assert np.allclose(t, df.conv_trim_time_ms, atol=0.11)


def test_flow_iterations_rules():
    true = np.array([10, 10, 10])
    pred = np.array([10, 11, 14])
    assert flow_iterations(pred, true, 6).tolist() == [1, 2, 7]


def test_predict_codes_clipped_to_dac_range():
    class Const:
        def predict(self, X):
            return np.array([-5.0, 3.4, 99.0])

    assert predict_codes(Const(), None, 6).tolist() == [0, 3, 63]


def test_features_exclude_labels_and_truth(df):
    banned = {"latent_defect", "sort_pass", "final_pass", "fuse_map_hex"} | {t["code"] for t in TRIMS.values()}
    for name in TRIMS:
        for feats in feature_sets(name).values():
            assert not banned & set(feats)


def test_models_beat_median_baseline_on_holdout(df):
    a = df[df.derivative == "MCU-A"]
    tr, te = wafer_holdout(a)
    feats = feature_sets("bg")["base"]
    t = TRIMS["bg"]
    errs = {}
    for kind in ("median", "ridge"):
        m = make_model(kind).fit(tr[feats], tr[t["code"]])
        errs[kind] = np.abs(predict_codes(m, te[feats], t["bits"]) - te[t["code"]]).mean()
    assert errs["ridge"] < 1.0 < errs["median"]


def test_every_model_input_is_a_dictionary_feature_column():
    """Dataset dictionary roles: only role=feature columns (plus values derived from them) may feed a model.
    Regression: HSI/LDO once used bg_trimmed_sort_V, which the dictionary marks as a label."""
    from anomaly import FEATS as ANOMALY_FEATS, MEASURES
    from data import DERIVED_FEATURES, FEATURE_COLUMNS
    from drift import FEATS as DRIFT_FEATS
    from ts_cal import FEATURES as TS_FEATURES

    allowed = FEATURE_COLUMNS | DERIVED_FEATURES
    used = set(ANOMALY_FEATS) | set(MEASURES) | set(DRIFT_FEATS)
    for feats in TS_FEATURES.values():
        used |= set(feats)
    for name in TRIMS:
        for feats in feature_sets(name).values():
            used |= set(feats)
    assert used <= allowed, f"label/meta columns used as features: {sorted(used - allowed)}"
    assert "bg_trimmed_sort_V" not in used


def _dirty(df):
    """Small frame with one example of every problem the cleaning step handles."""
    d = df.head(20).copy().reset_index(drop=True)
    d.loc[0, "derivative"] = "  MCU-A "                      # whitespace
    d.loc[1, "fuse_map_hex"] = d.loc[1, "fuse_map_hex"].lower().replace("0x", "0x")  # lower-case hex
    d.loc[2, "pcm_tox_A"] = np.nan                            # missing required value
    d.loc[3, "bg_trim_code_opt"] = 99                         # outside 6-bit DAC range
    d = pd.concat([d, d.iloc[[5]], d.iloc[[6]].assign(pcm_vth_n_mV=1.0)], ignore_index=True)  # exact dup + repeated die_id
    return d


def test_clean_mcu_fixes_every_problem_and_logs_it(df):
    from data import clean_mcu

    out, log = clean_mcu(_dirty(df))
    assert log["rows_in"] == 22 and log["rows_out"] == 18
    assert log["duplicate_rows_dropped"] == 1 and log["duplicate_die_id_dropped"] == 1
    assert log["rows_dropped_missing_required"] == 1 and log["missing_required_by_column"] == {"pcm_tox_A": 1}
    assert log["rows_dropped_code_out_of_range"] == 1
    assert (out.derivative == out.derivative.str.strip()).all()
    assert out.fuse_map_hex.str.fullmatch(r"0x[0-9A-F]{16}").all()
    assert not out.die_id.duplicated().any()
    validate_mcu(out)  # cleaned output passes the strict validator


def test_clean_mcu_is_a_no_op_on_the_shipped_file(df):
    from data import clean_mcu

    raw = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "mcu_trim_fuse.csv")
    out, log = clean_mcu(raw)
    assert out.equals(raw) and log["rows_in"] == log["rows_out"] == 8420
    assert log["duplicate_rows_dropped"] == log["rows_dropped_missing_required"] == log["rows_dropped_code_out_of_range"] == 0
    assert log["fuse_word_vs_trim_column_mismatches_kept"] == 0 and log["extreme_outlier_values_kept"] == 18


def test_clean_mcu_parses_text_booleans_and_bad_numbers(df):
    from data import clean_mcu

    d = df.head(10).copy()
    d["sort_pass"] = d["sort_pass"].map({True: "True", False: "False"})
    d["bg_untrim_V"] = d["bg_untrim_V"].astype(object)
    d.loc[4, "bg_untrim_V"] = "n/a"                          # unreadable number becomes missing, row is dropped
    out, log = clean_mcu(d)
    assert len(out) == 9 and log["missing_required_by_column"] == {"bg_untrim_V": 1}
    assert out["sort_pass"].dtype.name == "boolean"
