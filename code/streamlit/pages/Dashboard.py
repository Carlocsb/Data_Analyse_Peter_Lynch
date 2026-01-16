import streamlit as st
import pandas as pd
import sys
import os

# Path setup
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.funktionen import (
    get_es_connection,
    render_source_selector,
    search_stock_in_es,
    load_historical_metrics,
    plot_metric_history,
    calculate_peter_lynch_category,
    build_metrics_table,
    describe_metrics,
)

# === 1️⃣ Setup ===
st.set_page_config(page_title="Stock Search", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png", caption="")
st.title("🔍 Stock Search and Metrics Dashboard")
st.markdown("Enter a **ticker symbol** (e.g., AAPL, MSFT, NVDA):")

# === 2️⃣ Elasticsearch connection & data source toggle ===
es = get_es_connection()
source_mode = render_source_selector()  # Sidebar toggle for data source

# === 3️⃣ Input field ===
raw = st.text_input("", placeholder="e.g., AAPL or TSLA")
query_symbol = (raw or "").strip().upper()

# Styling for input field
st.markdown(
    """
<style>
    input {
        color: white;
        background-color: #1e1e1e;
    }
</style>
""",
    unsafe_allow_html=True,
)

# === 4️⃣ Main display ===
if query_symbol:
    data = search_stock_in_es(es, query_symbol, source_mode)  # pass mode

    if data:
        st.subheader(f"📊 Metrics for: {data.get('symbol','N/A')}")
        st.caption(
            f"Document source: {data.get('source','—')} • "
            f"Mode: {source_mode} • "
            f"Date: {data.get('date','N/A')}"
        )

        # === Key metrics ===
        col1, col2, col3 = st.columns(3)
        col1.metric("🏭 Industry", data.get("industry", "—"))
        col2.metric("💼 Sector", data.get("sector", "—"))
        col3.metric(
            "💰 Market Cap",
            f"{data.get('marketCap', 0)/1e9:.2f} Bn USD" if data.get("marketCap") else "—",
        )

        col4, col5, col6 = st.columns(3)
        col4.metric("📈 P/E Ratio", round(data.get("peRatio", 0), 2) if data.get("peRatio") else "—")
        col5.metric("🏦 Book Value / Share", round(data.get("bookValuePerShare", 0), 2) if data.get("bookValuePerShare") else "—")
        col6.metric("📉 Price / Book", round(data.get("priceToBook", 0), 2) if data.get("priceToBook") else "—")

        col7, col8, col9 = st.columns(3)
        col7.metric("💸 Dividend Yield", f"{data.get('dividendYield')*100:.2f} %" if data.get("dividendYield") else "—")
        col8.metric("📊 EPS", round(data.get("eps", 0), 2) if data.get("eps") else "—")
        col9.metric("⚖️ Debt / Equity", round(data.get("debtToEquity", 0), 2) if data.get("debtToEquity") else "—")

        # === 🧭 Peter Lynch categorization ===
        st.markdown("---")
        st.markdown("### 🧭 Peter Lynch Categorization")

        category_text, hit_rate = calculate_peter_lynch_category(data)

        st.success(
            f"🏷️ This stock most likely belongs to **{category_text}**, "
            f"because it meets **{hit_rate}%** of the criteria."
        )

        # === 📈 History of selected metrics ===
        st.markdown("---")
        st.markdown("### 📈 Selected Metrics History")

        colA, colB, colC = st.columns(3)
        if colA.button("Show P/E history"):
            df = load_historical_metrics(es, data["symbol"], "peRatio", source_mode)
            fig = plot_metric_history(df, data["symbol"], "P/E Ratio")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if colB.button("Show EPS history"):
            df = load_historical_metrics(es, data["symbol"], "eps", source_mode)
            fig = plot_metric_history(df, data["symbol"], "Earnings Per Share (EPS)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if colC.button("Show Price/Book history"):
            df = load_historical_metrics(es, data["symbol"], "priceToBook", source_mode)
            fig = plot_metric_history(df, data["symbol"], "Price / Book")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        colD, colE, colF = st.columns(3)
        if colD.button("Show dividend yield history"):
            df = load_historical_metrics(es, data["symbol"], "dividendYield", source_mode)
            fig = plot_metric_history(df, data["symbol"], "Dividend Yield", "%")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if colE.button("Show debt/equity history"):
            df = load_historical_metrics(es, data["symbol"], "debtToEquity", source_mode)
            fig = plot_metric_history(df, data["symbol"], "Debt / Equity")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if colF.button("Show free cash flow history"):
            df = load_historical_metrics(es, data["symbol"], "freeCashFlow", source_mode)
            fig = plot_metric_history(df, data["symbol"], "Free Cash Flow", "USD")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # === 🧩 Additional metrics ===
        st.markdown("---")
        st.markdown("### 🧩 Additional Metrics")
        df_tbl = build_metrics_table(data)
        st.dataframe(df_tbl, use_container_width=True)

        # === 📘 Descriptions of key metrics ===
        st.markdown("---")
        st.markdown("### 📘 Key Metrics Descriptions")
        for key, text in describe_metrics().items():
            if data.get(key) is not None:
                st.markdown(f"**{key}** – {text}")

else:
    st.info("🔎 Please enter a ticker symbol above (e.g., AAPL, MSFT, TSLA).")
