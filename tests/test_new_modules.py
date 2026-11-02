import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline_compare import cpk  # noqa: E402
from data import TRIMS, load_mcu, load_schema  # noqa: E402
from fuse_check import validate_word  # noqa: E402
from fuse_gen import PER_DIE, codes_from_row, generate_fuse_word  # noqa: E402
from transfer import predictions  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return load_mcu()


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def test_generator_reproduces_stored_words(df, schema):
    for r in df.sample(400, random_state=0).itertuples():
        assert generate_fuse_word(r.derivative, codes_from_row(r._asdict()), schema) == r.fuse_map_hex


def test_generated_words_pass_validator(df, schema):
    for _, r in df.sample(200, random_state=1).iterrows():
        w = generate_fuse_word(r.derivative, codes_from_row(r), schema)
        assert validate_word(w, r.derivative, schema, codes_from_row(r)) == []


@pytest.mark.parametrize("bad", [{"BG_TRIM": 64}, {"HSI_TRIM": 128}, {"LDO_TRIM": -1}, {"TS_CAL1": 4096}])
def test_generator_rejects_values_that_do_not_fit(df, schema, bad):
    r = df.iloc[0]
    with pytest.raises(ValueError, match="does not fit"):
        generate_fuse_word(r.derivative, {**codes_from_row(r), **bad}, schema)


def test_generator_rejects_unknown_derivative_and_missing_field(df, schema):
    r = df.iloc[0]
    with pytest.raises(ValueError, match="unknown derivative"):
        generate_fuse_word("MCU-Z", codes_from_row(r), schema)
    with pytest.raises(ValueError, match="exactly"):
        generate_fuse_word(r.derivative, {k: v for k, v in codes_from_row(r).items() if k != "BG_TRIM"}, schema)
    assert set(PER_DIE) == set(codes_from_row(r))


def test_cpk_known_values():
    x = np.array([1.199, 1.2, 1.201])  # mean 1.2, sd 0.001, spec +/-5 mV -> 5/3 = 1.667
    assert cpk(x, 1.2, 5e-3) == pytest.approx(5 / 3, rel=1e-6)
    assert cpk(x + 0.002, 1.2, 5e-3) < cpk(x, 1.2, 5e-3)  # off-centre lowers Cpk


def test_transfer_predictions_shapes_and_range(df):
    a = df[df.derivative == "MCU-A"]
    c = df[df.derivative == "MCU-C"]
    tune, test = c[c.wafer_id == c.wafer_id.iloc[0]].sample(25, random_state=0), c[c.wafer_id != c.wafer_id.iloc[0]]
    out = predictions("bg", a, tune, test)
    assert set(out) == {"zero_shot", "linear_fix", "pooled", "own_ridge"}
    for pred in out.values():
        assert len(pred) == len(test) and pred.min() >= 0 and pred.max() <= 2 ** TRIMS["bg"]["bits"] - 1
    assert set(predictions("bg", a, tune.iloc[:0], test)) == {"zero_shot"}  # n=0 is zero-shot only


def test_every_report_path_in_the_scripts_exists():
    """Reports are numbered in pipeline order; a renamed file without a matching script update would break this."""
    import anomaly, baseline_compare, drift, fuse_check, fuse_gen, transfer, trim_models, ts_cal  # noqa: E401

    for mod in (trim_models, drift, transfer, ts_cal, fuse_check, fuse_gen, anomaly, baseline_compare):
        assert mod.REPORT.exists(), f"{mod.__name__} writes {mod.REPORT.name}, which is not in reports/"
    root = Path(__file__).resolve().parents[1]
    assert (root / "reports" / "01_data_quality_report.md").exists()
    assert (root / "reports" / "00_implementation_summary.docx").exists()
