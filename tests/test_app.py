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
    assert "Synthetic data: demo_upload_with_issues.csv" in texts(at)  # recognised as generated data


# ---------------------------------------------------------------- cross-page behaviour (C1, C2, I1)
# These run several pages in ONE session, with the sidebar drawn before the page as in app.py, so that
# state carried between pages and reruns is tested (page() above runs a single page on its own).

APP_PAGES = """
import streamlit as st
from ui import components, state
from ui.navigation import sidebar
from ui.views import association_rules, bundles, data_explorer, evaluation_lab, recommendations, settings
components.inject_css()
state.init()
sidebar()
components.show_flash()
PAGES = {"rules": association_rules, "recommendations": recommendations, "bundles": bundles,
         "data": data_explorer, "evaluation": evaluation_lab, "settings": settings}
PAGES[st.session_state.get("test_page", "rules")].render()
"""


def session(page_name: str = "rules") -> AppTest:
    at = AppTest.from_string(APP_PAGES, default_timeout=TIMEOUT)
    at.session_state["test_page"] = page_name
    return at.run()


def go(at: AppTest, page_name: str) -> AppTest:
    at.session_state["test_page"] = page_name
    return at.run()


def sidebar_text(at: AppTest) -> str:
    return " ".join(str(e.value) for kind in ("markdown", "caption") for e in at.sidebar.get(kind))


def apply_support(at: AppTest, value: float) -> AppTest:
    labelled(at.slider, "Minimum support").set_value(value)
    labelled(at.button, "Apply settings").click()
    return at.run()


def test_consecutive_apply_uses_the_latest_values_and_refreshes_the_sidebar():
    at = session("rules")
    for value, shown in ((0.05, "5.0%"), (0.03, "3.0%"), (0.02, "2.0%")):
        apply_support(at, value)
        assert not at.exception
        assert at.session_state["config"].min_support == pytest.approx(value)
        assert labelled(at.slider, "Minimum support").value == pytest.approx(value)
        rules_on_page = {m.label: m.value for m in at.metric}["Association rules"]
        assert f"Support ≥ {shown}" in sidebar_text(at)
        assert f"**{rules_on_page}** association rules" in sidebar_text(at)  # sidebar and page agree
        assert "Mining settings applied." in texts(at)


def test_apply_keeps_the_other_settings_and_reports_invalid_ones():
    at = session("rules")
    labelled(at.slider, "Minimum confidence").set_value(0.5)
    labelled(at.multiselect, "Ignore products").set_value(["Milk"])
    labelled(at.button, "Apply settings").click()
    at.run()
    apply_support(at, 0.02)  # a second, separate apply must not undo the first one
    config = at.session_state["config"]
    assert (config.min_support, config.min_confidence, config.exclude) == (pytest.approx(0.02), 0.5, ("Milk",))


def test_the_basket_survives_moving_to_other_pages():
    at = session("recommendations")
    labelled(at.multiselect, "Search and add products").set_value(["Milk", "Bread"]).run()
    assert "1. Butter" in texts(at)
    for other in ("bundles", "rules", "settings"):
        go(at, other)
        assert not at.exception
    go(at, "recommendations")
    assert labelled(at.multiselect, "Search and add products").value == ["Milk", "Bread"]
    assert "1. Butter" in texts(at)
    labelled(at.multiselect, "Search and add products").set_value(["Pasta"]).run()  # changing it still works
    go(at, "bundles")
    go(at, "recommendations")
    assert labelled(at.multiselect, "Search and add products").value == ["Pasta"]
    labelled(at.button, "Clear basket").click().run()
    assert labelled(at.multiselect, "Search and add products").value == []


def test_ignored_products_never_appear_in_suggestions_bundles_or_insights():
    at = session("rules")
    labelled(at.multiselect, "Ignore products").set_value(["Milk", "Shampoo"])
    labelled(at.button, "Apply settings").click()
    at.run()
    assert "2 ignored" in sidebar_text(at)
    go(at, "recommendations")  # empty basket: best sellers only, and Milk is the top best seller
    titles = [m.value for m in at.markdown if "bw-card-title" in m.value and ". " in m.value]
    assert titles and not any("Milk" in title for title in titles)
    labelled(at.multiselect, "Search and add products").set_value(["Pasta"]).run()
    assert "Milk" not in " ".join(m.value for m in at.markdown if "bw-card-title" in m.value)
    go(at, "bundles")
    tables = [df.value for df in at.dataframe if "products" in df.value.columns]
    assert tables and not tables[0]["products"].str.contains("Milk|Shampoo").any()
    overview = AppTest.from_file(APP, default_timeout=TIMEOUT)
    overview.session_state["config"] = at.session_state["config"]
    overview.run()
    assert "Conditioner + Shampoo" not in texts(overview) and "Milk" in texts(overview)  # sales stats keep Milk


def test_a_refused_upload_keeps_its_error_visible_until_the_file_is_removed():
    at = session("data")
    at.file_uploader[0].upload("broken.csv", b"colour,size\nred,10\n", "text/csv").run()
    assert any("broken.csv could not be used" in e.value for e in at.error)
    labelled(at.number_input, "Transactions").set_value(1000).run()  # an unrelated interaction
    assert any("broken.csv could not be used" in e.value for e in at.error)
    assert {m.label: m.value for m in at.metric}["Valid transactions"] == "8,000"


def test_use_demo_data_when_it_is_already_active_resets_nothing():
    at = session("rules")
    labelled(at.multiselect, "Ignore products").set_value(["Milk"])
    labelled(at.button, "Apply settings").click()
    at.run()
    go(at, "data")
    labelled(at.button, "Use demo data").click().run()
    assert at.session_state["config"].exclude == ("Milk",)
    assert "already in use" in texts(at)


def real_world_page(path, rolling_path=None) -> AppTest:
    """The Evaluation Lab with the real-world tab pointed at `path` (a fixture instead of the downloaded file)
    and, if given, at `rolling_path` instead of the published multi-cutoff results."""
    script = ("import streamlit as st\nfrom ui import components, state\nfrom ui.views import evaluation_lab\n"
              "components.inject_css()\nstate.init()\nevaluation_lab.render()\n")
    at = AppTest.from_string(script, default_timeout=TIMEOUT)
    at.session_state["uci_path"] = str(path)
    if rolling_path is not None:
        at.session_state["rolling_path"] = str(rolling_path)
    return at.run()


def test_real_world_tab_explains_how_to_download_a_missing_dataset(tmp_path):
    at = real_world_page(tmp_path / "missing.csv")
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Random-split benchmark (active dataset)",
                                              "Real-world validation (UCI data, split by time)"]
    assert "has not been downloaded yet" in texts(at)
    assert any("download_uci_online_retail.py" in block.value for block in at.code)
    assert "CC BY 4.0" in texts(at) and "must not be combined" in texts(at)
    labelled(at.button, "Run evaluation").click().run()   # the random-split benchmark still works beside it
    assert not at.exception and any("hit rate@3" in df.value.columns for df in at.dataframe)


def test_real_world_tab_shows_published_multi_cutoff_results_without_running_anything(uci_csv, tmp_path):
    from market_basket import real_world, rolling

    source = real_world.prepare(uci_csv, progress=lambda _: None)
    result = rolling.evaluate(source, supports=(0.03,), confidences=(0.2,), lifts=(1.2,), n_bootstrap=50,
                              progress=lambda _: None)
    published = tmp_path / "rolling_folds.csv"
    result.folds.to_csv(published, index=False)
    at = real_world_page(tmp_path / "missing.csv", published)      # shown even before the dataset is downloaded
    assert not at.exception
    assert [e.label for e in at.expander if "monthly cutoffs" in e.label] == [
        "Results at 3 monthly cutoffs (published multi-cutoff check)"]
    table = next(df.value for df in at.dataframe if "difference (95% interval)" in df.value.columns)
    assert len(table) == 3 and table["test period"].iloc[0].startswith("2025-04-01 to ")
    assert table["association-rule recommender"].tolist() == result.folds["hit rate@3: " + rolling.PRIMARY].tolist()
    assert "uci-rolling" in texts(at) and "not independent" in texts(at)
    at = real_world_page(tmp_path / "missing.csv", tmp_path / "not_published.csv")
    assert not at.exception and not [e for e in at.expander if "monthly cutoffs" in e.label]


def test_real_world_tab_loads_runs_and_exports(uci_csv):
    at = real_world_page(uci_csv)
    assert not at.exception and "Results on the test period" not in texts(at)   # nothing runs before a click
    labelled(at.button, "Load and check the dataset").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Valid invoices"] == "1,500" and metrics["Raw rows"] == "5,716"
    assert any("step" in df.value.columns for df in at.dataframe)          # preprocessing summary
    labelled(at.button, "Run chronological evaluation").click().run()
    assert not at.exception
    comparison = next(df.value for df in at.dataframe if "Association-rule recommender" in df.value.columns)
    assert comparison["Metric"].tolist()[:4] == ["Hit Rate@3", "Precision@3", "Recall@3", "Coverage"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Best-seller Hit Rate@3"].endswith("%") and metrics["Association rules Hit Rate@3"].endswith("%")
    labels = [d.proto.label for d in at.get("download_button")]
    assert {"Download results (JSON)", "Download report (Markdown)", "Download all results (ZIP)"} <= set(labels)
    assert "Limitations" in [h.value for h in at.subheader]
    labelled(at.radio, "Test period").set_value("Fixed start date")
    labelled(at.button, "Run chronological evaluation").click().run()   # no date chosen: a clear message
    assert any("Choose the date" in e.value for e in at.error) and not at.exception


def test_evaluation_shows_the_settings_it_used_and_a_search_without_matches_says_so():
    at = session("evaluation")
    labelled(at.slider, "Random splits").set_value(1)
    labelled(at.button, "Run evaluation").click().run()
    assert "Result shown: 1 random split" in texts(at) and "FP-Growth" in texts(at)
    go(at, "rules")
    labelled(at.text_input, "Find rules mentioning a product").set_value("zzz-not-a-product").run()
    assert "No rule mentions" in texts(at) and not at.exception
