"""BasketWise: Retail Basket Intelligence. Run with: streamlit run app.py"""
import streamlit as st

from ui import components, state
from ui.navigation import PAGES, sidebar

st.set_page_config(page_title="BasketWise", page_icon=":material/shopping_basket:", layout="wide",
                   initial_sidebar_state="expanded")
components.inject_css()
state.init()

page = st.navigation(PAGES)
sidebar()
components.show_flash()
page.run()
