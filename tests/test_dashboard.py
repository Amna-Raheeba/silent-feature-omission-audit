import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_dashboard_runs_without_exception():
    at = AppTest.from_file(APP).run(timeout=180)
    assert not at.exception


def test_dashboard_has_three_tabs():
    at = AppTest.from_file(APP).run(timeout=180)
    assert len(at.tabs) == 3
