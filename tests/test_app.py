"""Interface tests: drive the Streamlit pages headlessly with streamlit.testing.AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from market_basket.data import load_demo_data
from market_basket.mining import mine_rules

TIMEOUT = 180
APP = str(Path(__file__).resolve().parent.parent / "app.py")
SMALL_CSV = ("transaction_id,product_name\n" + "".join(
    f"T{i},{a}\nT{i},{b}\n" for i, (a, b) in enumerate([("Milk", "Bread"), ("Milk", "Tea"), ("Bread", "Jam"),
                                                        ("Milk", "Bread"), ("Tea", "Sugar"), ("Milk", "Bread")])
)).encode()


def page(module: str, preload: bytes | None = None) -> AppTest:
    """Run one page on its own; optionally make an uploaded CSV the active dataset first."""
    preload_code = ""
    if preload is not None:
        preload_code = (f"if 'preloaded' not in st.session_state:\n"
                        f"    state.use_upload('small.csv', {preload!r})\n"
                        f"    st.session_state.preloaded = True\n")
    script = (
        "import streamlit as st\n"
        "from ui import components, state\n"
        f"from ui.views import {module}\n"
        "components.inject_css()\n"
        "state.init()\n"
        f"{preload_code}"
        "components.show_flash()\n"
        f"{module}.render()\n"
    )
    return AppTest.from_string(script, default_timeout=TIMEOUT).run()


def labelled(elements, label):
    return next(element for element in elements if element.label == label)


def texts(at) -> str:
    return " ".join(str(e.value) for kind in ("markdown", "caption", "info", "success", "warning", "error")
                    for e in at.get(kind))


def test_app_starts_on_the_overview_page():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "Overview"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Transactions"] == "8,000" and metrics["Product categories"] == "8"
    assert any("association rules" in m.value for m in at.sidebar.markdown)
    assert "most frequently purchased product" in texts(at)


def test_uploading_a_valid_csv_replaces_the_dataset_and_shows_warnings():
    at = page("data_explorer")
    upload = ("transaction_id,product_name\nT1,Milk\nT1,Bread\n,Eggs\nT2,Milk\nT2,Tea\nT3,Bread\nT3,Butter\n").encode()
    at.file_uploader[0].upload("sales.csv", upload, "text/csv").run()
    assert not at.exception
    assert "Loaded sales.csv: 3 valid transactions" in texts(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Valid transactions"] == "3" and metrics["Rows removed or rejected"] == "1"
    assert any("rejected or removed" in w.value for w in at.warning)


def test_uploading_an_invalid_csv_keeps_the_previous_dataset():
    at = page("data_explorer")
    at.file_uploader[0].upload("broken.csv", b"colour,size\nred,10\n", "text/csv").run()
    assert not at.exception
    assert any("could not be used" in e.value and "Could not recognise the columns" in e.value for e in at.error)
    assert {m.label: m.value for m in at.metric}["Valid transactions"] == "8,000"


def test_generating_sample_data():
    at = page("data_explorer")
    labelled(at.number_input, "Transactions").set_value(500)
    labelled(at.number_input, "Products").set_value(12)
    labelled(at.button, "Generate sample data").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Valid transactions"] == "500" and metrics["Unique products"] == "12"


def test_rule_settings_search_and_empty_state():
    at = page("association_rules")
    assert {m.label: m.value for m in at.metric}["Association rules"] != "0"
    labelled(at.text_input, "Find rules mentioning a product").set_value("pasta").run()
    assert "rules shown" in texts(at)
    labelled(at.slider, "Minimum support").set_value(0.1)
    labelled(at.slider, "Minimum confidence").set_value(1.0)
    at.button[0].click().run()  # form submit
    assert not at.exception
    assert "No rules pass these settings" in texts(at)


def test_recommendations_for_a_basket_and_an_empty_basket():
    at = page("recommendations")
    assert "The basket is empty" in texts(at)  # best sellers shown with an explanation
    labelled(at.multiselect, "Search and add products").set_value(["Milk", "Bread"]).run()
    assert not at.exception
    assert "1. Butter" in texts(at)
    assert "Customers whose basket contained Bread and Milk also bought Butter" in texts(at)


def test_basket_without_applicable_rules_and_no_fallback():
    rules = mine_rules(load_demo_data().transactions).rules
    in_rules = {p for s in rules["antecedents"] for p in s}
    lonely = next(p for p in load_demo_data().catalog if p not in in_rules)
    at = page("recommendations")
    labelled(at.toggle, "Fill free slots with best sellers").set_value(False)
    labelled(at.multiselect, "Search and add products").set_value([lonely]).run()
    assert not at.exception
    assert "No association rule applies to this basket" in texts(at)


def test_evaluation_lab_runs_and_reports_all_methods():
    at = page("evaluation_lab")
    assert "Run the evaluation" in texts(at)
    labelled(at.slider, "Random splits").set_value(1)
    at.button[0].click().run()
    assert not at.exception
    summary = next(df.value for df in at.dataframe if "hit rate@3" in df.value.columns)
    assert len(summary) == 4
    assert at.success or at.warning  # the computed interpretation


def test_bundles_and_settings_pages_render_with_downloads():
    bundles = page("bundles")
    assert not bundles.exception and len(bundles.dataframe) >= 2
    settings = page("settings")
    assert not settings.exception
    labels = [d.proto.label for d in settings.get("download_button")]
    assert "Download report (Markdown)" in labels and "Download all results (ZIP of CSV files)" in labels


@pytest.mark.parametrize("module", ["overview", "product_analytics", "association_rules", "recommendations",
                                    "bundles", "evaluation_lab", "settings"])
def test_pages_do_not_crash_on_a_tiny_dataset(module):
    if module == "overview":  # the overview links to other pages, so run it through the full app
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.session_state["source"] = {"key": "upload-small", "payload": SMALL_CSV, "name": "small.csv",
                                      "synthetic": False}
        at.run()
        assert {m.label: m.value for m in at.metric}["Transactions"] == "6"
    else:
        at = page(module, preload=SMALL_CSV)
    if module == "evaluation_lab":
        at.button[0].click().run()
    assert not at.exception


def test_uploading_the_demo_file_with_known_data_problems():
    from market_basket.config import PROJECT_ROOT

    at = page("data_explorer")
    payload = (PROJECT_ROOT / "Data" / "demo_upload_with_issues.csv").read_bytes()
    at.file_uploader[0].upload("demo_upload_with_issues.csv", payload, "text/csv").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Valid transactions"] == "2,997" and metrics["Unique products"] == "47"
    assert metrics["Rows removed or rejected"] == "275"  # 55 missing IDs + 110 blank names + 110 duplicates
    assert any("capitalisation" in w.value for w in at.warning)
