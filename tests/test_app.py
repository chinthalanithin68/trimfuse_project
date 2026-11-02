import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

import engine  # noqa: E402


def test_decide_priority_order():
    assert engine.decide(["DEBUG_LOCK"], 99, 0).startswith("BLOCK")  # fuse rule beats everything
    assert engine.decide([], 99, 0).startswith("HOLD")
    assert engine.decide([], 0, 0).startswith("PASS - predicted")
    assert engine.decide([], 0, 1).startswith("PASS - one")
    assert engine.decide([], 0, 5).startswith("FALLBACK")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception
    assert len(app.tabs) == 4


def test_app_fuse_checker_blocks_flipped_bit(app):
    app.number_input[[n.label for n in app.number_input].index("Flip one bit (-1 = none) to see which rule catches it")].set_value(41).run()
    assert not app.exception
    assert any("BLOCK" in e.value for e in app.error)


def test_app_fuse_checker_typed_word_branch():
    """Regression: the 'Type a word' branch has its own derivative selectbox and used to clash with the explorer's."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    at.radio[[r.label for r in at.radio].index("Start from")].set_value("Type a word").run()
    assert not at.exception
    at.checkbox[[c.label for c in at.checkbox].index("Also check against sort-log codes")].check().run()
    assert not at.exception


def test_app_fuse_generator_panel():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    at.number_input(key="gen_bg").set_value(63).run()
    assert not at.exception
    assert any("passes every rule" in s.value for s in at.success)
