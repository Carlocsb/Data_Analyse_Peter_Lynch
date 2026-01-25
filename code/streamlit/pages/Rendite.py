# pages/Rendite_Dreieck_und_SP500.py
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------------------------------------
# Pfad-Setup (Pages → src importierbar machen)
# ------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.funktionen import (
    get_es_connection,
    list_portfolios,
    load_portfolio,
)

from src.portfolio_simulation import (
    portfolio_doc_to_amounts,
    portfolio_doc_to_amount_weights,
    get_buy_date_like_dynamic_for_portfolio,
    build_saved_buyhold_series_with_liquidation,
)

try:
    from src.portfolio_simulation import simulate_dynamic_cached  # noqa
    HAS_DYNAMIC = True
except Exception:
    HAS_DYNAMIC = False

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PRICE_INDEX = os.getenv("ELASTICSEARCH_PRICE_INDEX", "prices")
BENCH_INDEX = os.getenv("ELASTICSEARCH_BENCH_INDEX", "benchmarks")
STOCKS_INDEX = os.getenv("ELASTICSEARCH_STOCKS_INDEX", "stocks")

GSPC_SYMBOL = os.getenv("GSPC_SYMBOL", "^GSPC")
SPXEW_SYMBOL = os.getenv("SPXEW_SYMBOL", "^SPXEW")

END_DATE = pd.Timestamp("2025-12-31")

st.set_page_config(page_title="Rendite-Dreieck & Benchmarks", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png")
st.title("Rendite-Dreieck & Benchmarks")

es = get_es_connection()

# ============================================================
# ES Helpers
# ============================================================
def es_fetch_all_by_symbol(index: str, symbol: str, source_fields: List[str]) -> pd.DataFrame:
    """Pagination via search_after; erwartet sort=date asc."""
    page_size = 5000
    body = {
        "size": page_size,
        "_source": source_fields,
        "query": {
            "bool": {
                "should": [
                    {"term": {"symbol": symbol}},
                    {"term": {"symbol.keyword": symbol}},
                ],
                "minimum_should_match": 1,
            }
        },
        "sort": [{"date": "asc"}],
    }

    rows = []
    resp = es.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    rows.extend([h.get("_source", {}) for h in hits])

    while hits:
        last_sort = hits[-1].get("sort")
        if not last_sort:
            break
        body["search_after"] = last_sort
        resp = es.search(index=index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        rows.extend([h.get("_source", {}) for h in hits])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "adjClose" in df.columns:
        df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
        df = df.dropna(subset=["date", "adjClose"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
    else:
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

    return df


def get_symbols_from_prices() -> List[str]:
    """Holt Symbol-Liste aus PRICE_INDEX per terms-aggregation."""
    for field in ["symbol.keyword", "symbol"]:
        try:
            body = {"size": 0, "aggs": {"symbols": {"terms": {"field": field, "size": 2000}}}}
            resp = es.search(index=PRICE_INDEX, body=body)
            buckets = resp.get("aggregations", {}).get("symbols", {}).get("buckets", [])
            syms = sorted([b["key"] for b in buckets])
            if syms:
                return syms
        except Exception:
            continue
    return []


@st.cache_data(show_spinner=True, ttl=300)
def load_prices(symbol: str) -> pd.DataFrame:
    return es_fetch_all_by_symbol(PRICE_INDEX, symbol, ["symbol", "date", "adjClose"])


@st.cache_data(show_spinner=True, ttl=300)
def load_gspc() -> pd.DataFrame:
    return es_fetch_all_by_symbol(BENCH_INDEX, GSPC_SYMBOL, ["symbol", "date", "adjClose"])


@st.cache_data(show_spinner=True, ttl=300)
def load_spxew() -> pd.DataFrame:
    return es_fetch_all_by_symbol(BENCH_INDEX, SPXEW_SYMBOL, ["symbol", "date", "adjClose"])


@st.cache_data(show_spinner=True, ttl=300)
def load_prices_matrix(symbols: List[str]) -> pd.DataFrame:
    """Preis-Matrix (index=date, cols=symbols). KEIN ffill (wichtig für Delisting-Logik)."""
    frames = []
    for sym in symbols:
        d = load_prices(sym)
        if d.empty:
            continue
        frames.append(d[["date", "adjClose"]].rename(columns={"adjClose": sym}).set_index("date"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ============================================================
# Rendite-Dreieck (Jahr) + Quartals-Dreieck
# ============================================================
def cagr(p0: float, p1: float, years: float) -> Optional[float]:
    if p0 <= 0 or p1 <= 0 or years <= 0:
        return None
    return (p1 / p0) ** (1.0 / years) - 1.0


def build_return_triangle(df: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    """df: date, adjClose"""
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["adjClose"] = pd.to_numeric(x["adjClose"], errors="coerce")
    x = x.dropna(subset=["date", "adjClose"]).sort_values("date")
    x["year"] = x["date"].dt.year

    eoy = x.groupby("year", as_index=False).tail(1)[["year", "date", "adjClose"]]
    eoy = eoy[(eoy["year"] >= year_from) & (eoy["year"] <= year_to)].copy()
    if eoy["year"].nunique() < 2:
        return pd.DataFrame()

    price_by_year = dict(zip(eoy["year"], eoy["adjClose"]))

    rows = []
    for buy in range(year_from, year_to + 1):
        for sell in range(year_from, year_to + 1):
            if sell <= buy:
                continue
            p0 = price_by_year.get(buy)
            p1 = price_by_year.get(sell)
            if p0 is None or p1 is None:
                continue
            y = sell - buy
            v = cagr(float(p0), float(p1), float(y))
            if v is None:
                continue
            rows.append({"buy_year": buy, "sell_year": sell, "cagr": v})
    return pd.DataFrame(rows)


def build_quarter_eoq_series(df: pd.DataFrame) -> pd.DataFrame:
    """df: date, adjClose → EOQ pro Quartal"""
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["adjClose"] = pd.to_numeric(x["adjClose"], errors="coerce")
    x = x.dropna(subset=["date", "adjClose"]).sort_values("date")

    x["year"] = x["date"].dt.year
    x["q"] = x["date"].dt.quarter
    x["quarter"] = x["year"].astype(str) + "-Q" + x["q"].astype(str)

    eoq = x.groupby(["year", "q"], as_index=False).tail(1)[["quarter", "date", "adjClose"]]
    return eoq.sort_values("date").reset_index(drop=True)


def build_quarter_triangle(df_like: pd.DataFrame, metric: str = "simple") -> pd.DataFrame:
    """df_like: quarter, ld, end_value"""
    if df_like is None or df_like.empty:
        return pd.DataFrame()

    df = df_like.copy()
    df["ld"] = pd.to_datetime(df["ld"], errors="coerce")
    df["end_value"] = pd.to_numeric(df["end_value"], errors="coerce")
    df = df.dropna(subset=["quarter", "ld", "end_value"])
    df = df[df["end_value"] > 0].sort_values("ld")

    q = df["quarter"].astype(str).tolist()
    ld = df["ld"].tolist()
    nav = df["end_value"].astype(float).to_numpy()

    rows = []
    n = len(q)
    for i in range(n):
        for j in range(n):
            if j <= i:
                continue

            nav_buy, nav_sell = nav[i], nav[j]
            if nav_buy <= 0 or nav_sell <= 0:
                continue

            if metric == "cagr":
                days = (ld[j] - ld[i]).days
                years = days / 365.25 if days and days > 0 else None
                if not years or years <= 0:
                    continue
                value = (nav_sell / nav_buy) ** (1.0 / years) - 1.0
            else:
                value = (nav_sell / nav_buy) - 1.0

            rows.append({"buy_q": q[i], "sell_q": q[j], "value": float(value)})

    return pd.DataFrame(rows)


def compute_benchmark_kpis(
    df_prices: pd.DataFrame,
    start_date: date,
    end_date: date,
    initial_capital: float,
) -> Dict[str, float]:
    """
    KPIs für Benchmark-Index wie Buy-&-Hold:
    - Skaliert Preisreihe so, dass Startwert = initial_capital
    - Keine Steuern (Steuer/Nettologik wird erst im UI angewandt)
    - Quartalsrenditen auf EOQ-Basis
    """
    if df_prices is None or df_prices.empty:
        return {"status": "empty"}

    x = df_prices.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["adjClose"] = pd.to_numeric(x["adjClose"], errors="coerce")
    x = x.dropna(subset=["date", "adjClose"]).sort_values("date")

    mask = (x["date"].dt.date >= start_date) & (x["date"].dt.date <= end_date)
    x = x.loc[mask].copy()
    if x.empty or len(x) < 2:
        return {"status": "too_few_points"}

    p0 = float(x["adjClose"].iloc[0])
    if p0 <= 0 or initial_capital <= 0:
        return {"status": "invalid_start"}

    scale = float(initial_capital) / p0
    series = x["adjClose"].astype(float) * scale

    end_value = float(series.iloc[-1])
    total_return = (end_value / float(initial_capital)) - 1.0

    d0 = pd.to_datetime(x["date"].iloc[0])
    d1 = pd.to_datetime(x["date"].iloc[-1])
    days = (d1 - d0).days
    years = days / 365.25 if days > 0 else float("nan")
    cagr_val = (end_value / float(initial_capital)) ** (1.0 / years) - 1.0 if years and years > 0 else float("nan")

    dd = (series / series.cummax()) - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    # Quartalsrenditen (EOQ) auf Preisbasis (unskaliert ok, Renditen sind invariant)
    eoq = build_quarter_eoq_series(x[["date", "adjClose"]])
    if eoq.empty or len(eoq) < 2:
        best_q = worst_q = avg_q = float("nan")
    else:
        eoq = eoq.sort_values("date").reset_index(drop=True)
        q_ret = eoq["adjClose"].pct_change()
        best_q = float(q_ret.max())
        worst_q = float(q_ret.min())
        avg_q = float(q_ret.mean())

    return {
        "status": "ok",
        "start_dt": d0,
        "end_dt": d1,
        "end_value": end_value,
        "total_return": total_return,
        "cagr": cagr_val,
        "max_dd": max_dd,
        "best_q": best_q,
        "worst_q": worst_q,
        "avg_q": avg_q,
    }


def plot_quarter_triangle(tri: pd.DataFrame, metric: str = "simple") -> alt.Chart:
    title = "Rendite" if metric == "simple" else "CAGR p.a."
    fmt_tooltip = ".2%"
    fmt_text = ".1%"

    sell_order = sorted(tri["sell_q"].unique(), reverse=True)
    buy_order = sorted(tri["buy_q"].unique(), reverse=True)

    heat = (
        alt.Chart(tri)
        .mark_rect()
        .encode(
            x=alt.X("sell_q:O", title="Verkauf (Quartal)", sort=sell_order),
            y=alt.Y("buy_q:O", title="Kauf (Quartal)", sort=buy_order),
            color=alt.Color("value:Q", title=title, scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=[
                alt.Tooltip("buy_q:O", title="Kauf"),
                alt.Tooltip("sell_q:O", title="Verkauf"),
                alt.Tooltip("value:Q", title=title, format=fmt_tooltip),
            ],
        )
        .properties(height=520)
    )

    text = (
        alt.Chart(tri)
        .mark_text(baseline="middle", fontSize=9)
        .encode(
            x=alt.X("sell_q:O", sort=sell_order),
            y=alt.Y("buy_q:O", sort=buy_order),
            text=alt.Text("value:Q", format=fmt_text),
        )
    )

    return heat + text


def build_quarter_triangle_from_prices(df_daily: pd.DataFrame, metric: str = "simple") -> pd.DataFrame:
    eoq = build_quarter_eoq_series(df_daily)
    if eoq.empty:
        return pd.DataFrame()
    df_like = eoq.rename(columns={"date": "ld", "adjClose": "end_value"})[["quarter", "ld", "end_value"]]
    return build_quarter_triangle(df_like, metric=metric)


# ============================================================
# Saved Portfolio: Summary + Detail (Ranking & Details)
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def compute_saved_portfolio_detail(pid: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    doc = load_portfolio(es, pid)
    if not doc:
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing_doc", "name": ""}

    weights = portfolio_doc_to_amount_weights(doc)
    if not weights:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_weights", "name": doc.get("name", "")}

    buy_dt = get_buy_date_like_dynamic_for_portfolio(es, STOCKS_INDEX, doc)
    if buy_dt is None:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_buy_dt", "name": doc.get("name", "")}

    syms = sorted(weights.keys())
    prices_mat = load_prices_matrix(syms)
    if prices_mat.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_prices", "name": doc.get("name", "")}

    df_nav, df_trades = build_saved_buyhold_series_with_liquidation(
        prices=prices_mat,
        weights=weights,
        buy_date=buy_dt,
        initial_capital=1000.0,
        end_date=END_DATE,
    )
    if df_nav.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "nav_empty", "name": doc.get("name", "")}

    start_value = 1000.0
    end_value = float(df_nav["nav"].iloc[-1])
    total_return = (end_value / start_value) - 1.0

    d0 = pd.to_datetime(df_nav["date"].iloc[0])
    d1 = pd.to_datetime(df_nav["date"].iloc[-1])
    days = (d1 - d0).days
    years = days / 365.25 if days > 0 else float("nan")
    cagr_val = (end_value / start_value) ** (1.0 / years) - 1.0 if years and years > 0 else float("nan")

    nav = df_nav["nav"].astype(float)
    dd = (nav / nav.cummax()) - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    meta = {
        "status": "ok",
        "name": doc.get("name", ""),
        "buy_dt": d0,
        "end_dt": d1,
        "end_value": end_value,
        "total_return": total_return,
        "cagr": cagr_val,
        "max_dd": max_dd,
    }
    return df_nav, df_trades, meta


@st.cache_data(show_spinner=True, ttl=600)
def compute_all_saved_portfolio_summaries(portfolios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in portfolios:
        pid = p.get("id")
        if not pid:
            continue
        _, _, meta = compute_saved_portfolio_detail(pid)
        rows.append({
            "id": pid,
            "name": meta.get("name") or p.get("name", ""),
            "status": meta.get("status", "unknown"),
            "end_value": meta.get("end_value", float("nan")),
            "total_return": meta.get("total_return", float("nan")),
            "cagr": meta.get("cagr", float("nan")),
            "max_dd": meta.get("max_dd", float("nan")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["end_value", "total_return", "cagr", "max_dd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# Sidebar UI (Reihenfolge wie gewünscht)
# ============================================================
rank_metric = "CAGR p.a."  # Default für Saved-Portfolios

with st.sidebar:
    st.header("Einstellungen")

    mode = st.radio(
        "Rendite-Dreieck Quelle",
        ["Einzeltitel", "Gespeichertes Portfolio", "Dynamisch (Quartals-Umstieg)"],
        index=0,
    )

    # 1) Portfolio-Details ganz oben (ohne Tabelle)
    selected_pid: Optional[str] = None
    symbol: Optional[str] = None

    if mode == "Gespeichertes Portfolio":
        portfolios = list_portfolios(es)
        if not portfolios:
            st.warning("Keine gespeicherten Portfolios gefunden.")
        else:
            labels = ["—"] + [f'{p.get("name","(ohne Name)")} — {p.get("id")}' for p in portfolios]
            sel = st.selectbox("Portfolio (Details)", labels, index=0)
            if sel != "—":
                idx = labels.index(sel) - 1
                selected_pid = portfolios[idx].get("id")

        rank_metric = st.radio(
            "Ranking-Metrik (Saved Portfolios)",
            ["CAGR p.a.", "Gesamtrendite"],
            index=0,
            horizontal=True,
        )

    # 2) Modus-spezifische Controls
    triangle_view = "Jahre (CAGR p.a.)"
    year_from = 2010
    year_to = 2025

    tax_rate = 0.2
    show_debug_table = True
    show_charts = True

    if mode == "Einzeltitel":
        symbols = get_symbols_from_prices()
        if not symbols:
            st.warning(f"Keine Symbole im Index '{PRICE_INDEX}' gefunden.")
        else:
            default_sym = "AAPL" if "AAPL" in symbols else symbols[0]
            symbol = st.selectbox("Aktie", symbols, index=symbols.index(default_sym))

        st.divider()
        st.caption("Dreieck-Ansicht")
        triangle_view = st.radio(
            "Dreieck basiert auf ...",
            ["Jahre (CAGR p.a.)", "Quartale (kumuliert)", "Quartale (CAGR p.a.)"],
            index=0,
        )

        st.divider()
        st.caption("Zeitraum (Dreieck)")
        year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
        year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)

    elif mode == "Dynamisch (Quartals-Umstieg)":
        st.caption("Dynamik: nutzt Quartals-Portfolios (YYYY-Qx) aus ES.")

        st.divider()
        st.caption("Dynamische Strategie (Quartal)")
        tax_rate = st.number_input(
            "Steuersatz auf Gewinne (z.B. 0.2 = 20%)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
        )
        show_charts = st.checkbox("Charts anzeigen", value=True)
        show_debug_table = st.checkbox("Ergebnis-Tabelle anzeigen", value=True)

        st.divider()
        st.caption("Dreieck-Ansicht")
        triangle_view = st.radio(
            "Dreieck basiert auf ...",
            ["Jahre (CAGR p.a.)", "Quartale (kumuliert)", "Quartale (CAGR p.a.)"],
            index=0,
        )

        st.divider()
        st.caption("Zeitraum (Dreieck & Dynamik)")
        year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
        year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)

    # 3) Benchmarks als vorletztes
    st.divider()
    st.caption("Benchmarks")

    show_bench = st.checkbox("Benchmarks anzeigen (GSPC & SPXEW)", value=True)
    if show_bench:
        with st.expander("Benchmark Zeitraum", expanded=False):
            b_from = st.date_input("Von Datum", value=date(2010, 1, 1))
            b_to = st.date_input("Bis Datum", value=date.today())
    else:
        b_from = date(2010, 1, 1)
        b_to = date.today()

    bench_view = "Brutto"
    if show_bench:
        bench_view = st.radio(
            "Benchmark Anzeige",
            ["Brutto", "Netto (20% Steuer auf Gewinn)"],
            index=0,
            horizontal=True,
        )

    # 4) Startkapital ganz nach unten (letztes)
    st.divider()
    initial_investment = st.number_input("Startkapital (USD)", min_value=0.0, value=1000.0, step=100.0)


# ============================================================
# Main: Auswertung pro Modus
# ============================================================
st.subheader("Auswertung")

# -----------------------------
# A) Gespeichertes Portfolio: Ranking-Bars + Details (kein Dreieck!)
# -----------------------------
if mode == "Gespeichertes Portfolio":
    st.subheader("Gespeicherte Portfolios – Übersicht (Buy-&-Hold)")

    portfolios = list_portfolios(es)
    if not portfolios:
        st.warning("Keine gespeicherten Portfolios gefunden.")
        st.stop()

    df_sum = compute_all_saved_portfolio_summaries(portfolios)
    df_ok = df_sum[df_sum["status"] == "ok"].copy()

    if df_ok.empty:
        st.warning("Keine gültigen Portfolios (prüfe: Name=YYYY-Qx, stocks/prices vorhanden).")
        st.dataframe(df_sum[["name", "id", "status"]], use_container_width=True, hide_index=True)
        st.stop()

    # --- Ranking-Metrik wählen (CAGR oder Gesamtrendite) ---
    use_cagr = (rank_metric == "CAGR p.a.")
    if use_cagr:
        df_ok["rank_val"] = df_ok["cagr"]
        x_title = "CAGR p.a. (%)"
        tip_title = "CAGR p.a."
    else:
        df_ok["rank_val"] = df_ok["total_return"]
        x_title = "Gesamtrendite (%)"
        tip_title = "Gesamtrendite"

    df_ok["rank_pct"] = df_ok["rank_val"] * 100.0
    df_ok = df_ok.sort_values("rank_pct", ascending=True)

    bars = (
        alt.Chart(df_ok)
        .mark_bar()
        .encode(
            y=alt.Y("name:N", sort=None, title="Portfolio"),
            x=alt.X("rank_pct:Q", title=x_title),
            tooltip=[
                alt.Tooltip("name:N", title="Portfolio"),
                alt.Tooltip("rank_pct:Q", title=tip_title, format=".2f"),
                alt.Tooltip("end_value:Q", title="Endbetrag", format=",.2f"),
                alt.Tooltip("cagr:Q", title="CAGR p.a.", format=".2%"),
                alt.Tooltip("total_return:Q", title="Gesamtrendite", format=".2%"),
                alt.Tooltip("max_dd:Q", title="Max Drawdown", format=".2%"),
            ],
        )
        .properties(height=max(320, min(900, 18 * len(df_ok))))
    )
    st.altair_chart(bars, use_container_width=True)

    st.divider()

    # Default-Selection: bestes Portfolio nach gewählter Metrik
    if not selected_pid:
        selected_pid = str(df_ok.sort_values("rank_pct", ascending=False).iloc[0]["id"])
        st.info("Kein Portfolio ausgewählt – zeige automatisch das beste Portfolio aus der Übersicht.")

    df_nav, df_trades, meta = compute_saved_portfolio_detail(selected_pid)
    if meta.get("status") != "ok" or df_nav.empty:
        st.warning(f"Details konnten nicht geladen werden. Status: {meta.get('status')}")
        st.stop()

    st.subheader("Ausgewähltes Portfolio – Details (Buy-&-Hold, Verkauf am Ende)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Endbetrag", f"{float(meta['end_value']):,.2f} USD")
    k2.metric("Gesamtrendite", f"{float(meta['total_return'])*100:.1f}%")
    k3.metric("CAGR p.a.", f"{float(meta['cagr'])*100:.1f}%")
    k4.metric("Max Drawdown", f"{float(meta['max_dd'])*100:.1f}%")

    st.caption(
        f"Kaufdatum wie Dynamik (ld → nächster Handelstag): {pd.to_datetime(meta['buy_dt']).date()} | "
        f"Ende: letzter verfügbarer Handelstag ≤ 31.12.2025 je Aktie (Liquidation in Cash)."
    )

    with st.expander("Dokumentation: Trades / Liquidationen", expanded=True):
        t = df_trades.copy()
        t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce")
        st.dataframe(t, use_container_width=True, hide_index=True)

# -----------------------------
# B) Einzeltitel: unverändert
# -----------------------------
elif mode == "Einzeltitel":
    if symbol is None:
        st.stop()

    df_base = load_prices(symbol)
    if df_base.empty:
        st.warning("Keine Preisdaten gefunden.")
        st.stop()

    st.subheader("Rendite-Dreieck")

    if triangle_view.startswith("Quartale"):
        metric = "cagr" if "CAGR" in triangle_view else "simple"
        tri_q = build_quarter_triangle_from_prices(df_base, metric=metric)
        if tri_q.empty:
            st.warning("Quartals-Rendite-Dreieck konnte nicht berechnet werden.")
        else:
            st.altair_chart(plot_quarter_triangle(tri_q, metric=metric), use_container_width=True)
    else:
        df_base["date"] = pd.to_datetime(df_base["date"], errors="coerce")
        min_y = int(df_base["date"].dt.year.min())
        max_y = int(df_base["date"].dt.year.max())
        y0 = max(int(year_from), min_y)
        y1 = min(int(year_to), max_y)

        if y1 - y0 < 2:
            st.warning(f"Zu wenig Daten für {y0}–{y1}. Bitte Zeitraum erweitern.")
        else:
            tri = build_return_triangle(df_base, y0, y1)
            if tri.empty:
                st.warning("Rendite-Dreieck konnte nicht berechnet werden.")
            else:
                heat = (
                    alt.Chart(tri)
                    .mark_rect()
                    .encode(
                        x=alt.X("sell_year:O", title="Verkauf (Jahr)", sort="descending"),
                        y=alt.Y("buy_year:O", title="Kauf (Jahr)", sort="descending"),
                        color=alt.Color("cagr:Q", title="CAGR p.a.", scale=alt.Scale(scheme="redyellowgreen")),
                        tooltip=[
                            alt.Tooltip("buy_year:O", title="Kauf"),
                            alt.Tooltip("sell_year:O", title="Verkauf"),
                            alt.Tooltip("cagr:Q", title="CAGR p.a.", format=".2%"),
                        ],
                    )
                    .properties(height=520)
                )
                text = (
                    alt.Chart(tri)
                    .mark_text(baseline="middle", fontSize=11)
                    .encode(
                        x=alt.X("sell_year:O", sort="descending"),
                        y=alt.Y("buy_year:O", sort="descending"),
                        text=alt.Text("cagr:Q", format=".1%"),
                    )
                )
                st.altair_chart(heat + text, use_container_width=True)

# -----------------------------
# C) Dynamisch: unverändert (nutzt src.simulate_dynamic_cached)
# -----------------------------
else:
    if not HAS_DYNAMIC:
        st.error(
            "Dynamik ist aktuell nicht verfügbar, weil `simulate_dynamic_cached` nicht in `src/portfolio_simulation.py` gefunden wurde.\n\n"
            "Hinweis: Stelle sicher, dass die Funktion dort existiert und bei @st.cache_data der ES-Client als `_es_client` (mit Unterstrich) übergeben wird."
        )
        st.stop()

    portfolios = list_portfolios(es)
    if not portfolios:
        st.warning("Keine gespeicherten Portfolios gefunden.")
        st.stop()

    minimal = [{"id": p.get("id"), "name": p.get("name", "")} for p in portfolios if p.get("id")]
    df_dyn = simulate_dynamic_cached(
        _es_client=es,  # wichtig: underscore, damit Streamlit nicht hasht
        portfolio_minimal=minimal,
        initial_capital=float(initial_investment),
        tax_rate=float(tax_rate),
        year_from=int(year_from),
        year_to=int(year_to),
        prices_index=PRICE_INDEX,
        stocks_index=STOCKS_INDEX,
    )

    if df_dyn.empty:
        st.warning("Keine dynamischen Ergebnisse (prüfe stocks/prices Daten oder Zeitraum).")
        st.stop()

    # --- Dreieck für Dynamik ---
    st.subheader("Rendite-Dreieck (Dynamik)")
    if triangle_view.startswith("Quartale"):
        metric = "cagr" if "CAGR" in triangle_view else "simple"
        tri_q = build_quarter_triangle(df_dyn, metric=metric)
        if tri_q.empty:
            st.warning("Quartals-Rendite-Dreieck konnte nicht berechnet werden.")
        else:
            st.altair_chart(plot_quarter_triangle(tri_q, metric=metric), use_container_width=True)
    else:
        df_base_year = df_dyn[["ld", "end_value"]].rename(columns={"ld": "date", "end_value": "adjClose"}).copy()
        df_base_year["date"] = pd.to_datetime(df_base_year["date"], errors="coerce")
        df_base_year["adjClose"] = pd.to_numeric(df_base_year["adjClose"], errors="coerce")
        df_base_year = df_base_year.dropna(subset=["date", "adjClose"])

        min_y = int(df_base_year["date"].dt.year.min())
        max_y = int(df_base_year["date"].dt.year.max())
        y0 = max(int(year_from), min_y)
        y1 = min(int(year_to), max_y)

        tri = build_return_triangle(df_base_year, y0, y1)
        if tri.empty:
            st.warning("Jahres-Rendite-Dreieck konnte nicht berechnet werden.")
        else:
            heat = (
                alt.Chart(tri)
                .mark_rect()
                .encode(
                    x=alt.X("sell_year:O", title="Verkauf (Jahr)", sort="descending"),
                    y=alt.Y("buy_year:O", title="Kauf (Jahr)", sort="descending"),
                    color=alt.Color("cagr:Q", title="CAGR p.a.", scale=alt.Scale(scheme="redyellowgreen")),
                    tooltip=[
                        alt.Tooltip("buy_year:O", title="Kauf"),
                        alt.Tooltip("sell_year:O", title="Verkauf"),
                        alt.Tooltip("cagr:Q", title="CAGR p.a.", format=".2%"),
                    ],
                )
                .properties(height=520)
            )
            text = (
                alt.Chart(tri)
                .mark_text(baseline="middle", fontSize=11)
                .encode(
                    x=alt.X("sell_year:O", sort="descending"),
                    y=alt.Y("buy_year:O", sort="descending"),
                    text=alt.Text("cagr:Q", format=".1%"),
                )
            )
            st.altair_chart(heat + text, use_container_width=True)

    # --- KPIs/Charts wie vorher ---
    st.subheader("Dynamische Strategie – Ergebnisse")

    endbetrag = float(df_dyn.iloc[-1]["end_value"])
    total_return = (endbetrag / float(initial_investment) - 1.0) if initial_investment > 0 else float("nan")

    days = (df_dyn["ld"].iloc[-1] - df_dyn["ld"].iloc[0]).days
    years = days / 365.25 if days > 0 else float("nan")
    cagr_val = (
        (endbetrag / float(initial_investment)) ** (1.0 / years) - 1.0
        if initial_investment > 0 and years and years > 0
        else float("nan")
    )

    nav = df_dyn["end_value"].astype(float)
    roll_max = nav.cummax()
    dd = (nav / roll_max) - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    best_q = float(df_dyn["return_q"].max())
    worst_q = float(df_dyn["return_q"].min())
    avg_q = float(df_dyn["return_q"].mean())
    sum_tax = float(df_dyn["tax"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Endbetrag", f"{endbetrag:,.2f} USD")
    k2.metric("Gesamtrendite", f"{total_return*100:.1f}%")
    k3.metric("CAGR p.a.", f"{cagr_val*100:.1f}%")
    k4.metric("Max Drawdown", f"{max_dd*100:.1f}%")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Bestes Quartal", f"{best_q*100:.2f}%")
    k6.metric("Schlechtestes Quartal", f"{worst_q*100:.2f}%")
    k7.metric("Ø Quartalsrendite", f"{avg_q*100:.2f}%")
    k8.metric("Summe Steuer", f"{sum_tax:,.2f} USD")

    if show_charts:
        nav_line = (
            alt.Chart(df_dyn)
            .mark_line()
            .encode(
                x=alt.X("ld:T", title="Stichtag (ld)"),
                y=alt.Y("end_value:Q", title="NAV / Endbetrag"),
                tooltip=[
                    alt.Tooltip("quarter:N", title="Quartal"),
                    alt.Tooltip("ld:T", title="ld"),
                    alt.Tooltip("end_value:Q", title="Endbetrag", format=",.2f"),
                    alt.Tooltip("tax:Q", title="Steuer", format=",.2f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(nav_line, use_container_width=True)

        df_ret = df_dyn.dropna(subset=["return_q"]).copy()
        df_ret["return_pct"] = df_ret["return_q"] * 100.0
        ret_bar = (
            alt.Chart(df_ret)
            .mark_bar()
            .encode(
                x=alt.X("quarter:N", title="Quartal", sort=None),
                y=alt.Y("return_pct:Q", title="Rendite (%)"),
                tooltip=[
                    alt.Tooltip("quarter:N", title="Quartal"),
                    alt.Tooltip("return_pct:Q", title="Rendite", format=".2f"),
                    alt.Tooltip("tax:Q", title="Steuer", format=",.2f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(ret_bar, use_container_width=True)

        tax_bar = (
            alt.Chart(df_dyn)
            .mark_bar()
            .encode(
                x=alt.X("quarter:N", title="Quartal", sort=None),
                y=alt.Y("tax:Q", title="Steuer (USD)"),
                tooltip=[alt.Tooltip("quarter:N", title="Quartal"), alt.Tooltip("tax:Q", title="Steuer", format=",.2f")],
            )
            .properties(height=180)
        )
        st.altair_chart(tax_bar, use_container_width=True)

        df_dd = df_dyn.copy()
        df_dd["dd"] = (df_dd["end_value"] / df_dd["end_value"].cummax()) - 1.0
        dd_line = (
        alt.Chart(df_dd)
        .mark_line()
        .encode(
            x=alt.X("ld:T", title="Stichtag (ld)"),
            y=alt.Y("dd:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("quarter:N", title="Quartal"),
                alt.Tooltip("dd:Q", title="Drawdown", format=".2%"),
            ],
        )
                .properties(height=180)
        )
        st.altair_chart(dd_line, use_container_width=True)



    if show_debug_table:
        show_cols = [
            "quarter", "sell_date", "buy_date", "ld",
            "start_value", "gross_value", "tax", "end_value", "return_q",
            "n_old", "n_new",
            "old_missing_px_syms", "new_missing_px_syms",
            "new_target_syms", "new_bought_syms",
        ]
        df_show = df_dyn[show_cols].copy()
        df_show["return_q"] = df_show["return_q"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        st.dataframe(df_show, use_container_width=True, hide_index=True)


# ============================================================
# Benchmarks (optional, für alle Modi)
# ============================================================
if show_bench:
    st.divider()
    st.subheader("Benchmarks (Linien)")

    # -----------------------------
    # S&P 500 (GSPC)
    # -----------------------------
    with st.expander("S&P 500 (GSPC)", expanded=False):
        df_g = load_gspc()
        if df_g.empty:
            st.warning(f"Keine Daten für {GSPC_SYMBOL} im Index '{BENCH_INDEX}' gefunden.")
        else:
            kpi_g = compute_benchmark_kpis(
                df_prices=df_g,
                start_date=b_from,
                end_date=b_to,
                initial_capital=float(initial_investment),
            )

            if kpi_g.get("status") != "ok":
                st.warning("Keine ausreichenden Daten im gewählten Zeitraum für KPI-Berechnung.")
            else:
                # --- Brutto Basis
                end_brutto = float(kpi_g["end_value"])
                gain = end_brutto - float(initial_investment)

                # --- Netto-Logik nur wenn ausgewählt
                tax_rate_bench = 0.20
                is_netto = bench_view.startswith("Netto")
                bench_tax = max(0.0, gain) * tax_rate_bench if is_netto else 0.0
                end_netto = end_brutto - bench_tax

                end_display = end_netto if is_netto else end_brutto
                total_return_display = (end_display / float(initial_investment)) - 1.0 if initial_investment > 0 else float("nan")

                d0 = pd.to_datetime(kpi_g["start_dt"])
                d1 = pd.to_datetime(kpi_g["end_dt"])
                days = (d1 - d0).days
                years = days / 365.25 if days > 0 else float("nan")
                cagr_display = (
                    (end_display / float(initial_investment)) ** (1.0 / years) - 1.0
                    if initial_investment > 0 and years and years > 0
                    else float("nan")
                )

                # KPI Tiles
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Endbetrag ({'netto' if is_netto else 'brutto'})", f"{end_display:,.2f} USD")
                k2.metric("Gesamtrendite", f"{total_return_display*100:.1f}%")
                k3.metric("CAGR p.a.", f"{cagr_display*100:.1f}%")
                k4.metric("Max Drawdown", f"{float(kpi_g['max_dd'])*100:.1f}%")

                k5, k6, k7, k8 = st.columns(4)
                k5.metric("Bestes Quartal", f"{float(kpi_g['best_q'])*100:.2f}%")
                k6.metric("Schlechtestes Quartal", f"{float(kpi_g['worst_q'])*100:.2f}%")
                k7.metric("Ø Quartalsrendite", f"{float(kpi_g['avg_q'])*100:.2f}%")
                k8.metric("Steuer (20%)", f"{bench_tax:,.2f} USD" if is_netto else "—")

                # Optional: Brutto/Netto Zusatzinfo
                st.caption(
                    f"Zeitraum: {d0.date()} bis {d1.date()} | "
                    f"Skalierung: Startwert = {float(initial_investment):,.2f} USD"
                    + (" | Netto: 20% Steuer auf Gewinn (Endverkauf)" if is_netto else "")
                )

            # Chart (immer adjClose im Zeitraum)
            mask = (df_g["date"].dt.date >= b_from) & (df_g["date"].dt.date <= b_to)
            g = df_g.loc[mask].copy()
            if g.empty:
                st.warning("Keine Daten im gewählten Zeitraum.")
            else:
                st.altair_chart(
                    alt.Chart(g)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Datum"),
                        y=alt.Y("adjClose:Q", title="adjClose"),
                        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("adjClose:Q", format=",.2f")],
                    )
                    .properties(height=360),
                    use_container_width=True,
                )

    # -----------------------------
    # S&P 500 Equal Weight (SPXEW)
    # -----------------------------
    with st.expander("S&P 500 Equal Weight (SPXEW)", expanded=False):
        df_e = load_spxew()
        if df_e.empty:
            st.warning(f"Keine Daten für {SPXEW_SYMBOL} im Index '{BENCH_INDEX}' gefunden.")
        else:
            kpi_e = compute_benchmark_kpis(
                df_prices=df_e,
                start_date=b_from,
                end_date=b_to,
                initial_capital=float(initial_investment),
            )

            if kpi_e.get("status") != "ok":
                st.warning("Keine ausreichenden Daten im gewählten Zeitraum für KPI-Berechnung.")
            else:
                end_brutto = float(kpi_e["end_value"])
                gain = end_brutto - float(initial_investment)

                tax_rate_bench = 0.20
                is_netto = bench_view.startswith("Netto")
                bench_tax = max(0.0, gain) * tax_rate_bench if is_netto else 0.0
                end_netto = end_brutto - bench_tax

                end_display = end_netto if is_netto else end_brutto
                total_return_display = (end_display / float(initial_investment)) - 1.0 if initial_investment > 0 else float("nan")

                d0 = pd.to_datetime(kpi_e["start_dt"])
                d1 = pd.to_datetime(kpi_e["end_dt"])
                days = (d1 - d0).days
                years = days / 365.25 if days > 0 else float("nan")
                cagr_display = (
                    (end_display / float(initial_investment)) ** (1.0 / years) - 1.0
                    if initial_investment > 0 and years and years > 0
                    else float("nan")
                )

                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Endbetrag ({'netto' if is_netto else 'brutto'})", f"{end_display:,.2f} USD")
                k2.metric("Gesamtrendite", f"{total_return_display*100:.1f}%")
                k3.metric("CAGR p.a.", f"{cagr_display*100:.1f}%")
                k4.metric("Max Drawdown", f"{float(kpi_e['max_dd'])*100:.1f}%")

                k5, k6, k7, k8 = st.columns(4)
                k5.metric("Bestes Quartal", f"{float(kpi_e['best_q'])*100:.2f}%")
                k6.metric("Schlechtestes Quartal", f"{float(kpi_e['worst_q'])*100:.2f}%")
                k7.metric("Ø Quartalsrendite", f"{float(kpi_e['avg_q'])*100:.2f}%")
                k8.metric("Steuer (20%)", f"{bench_tax:,.2f} USD" if is_netto else "—")

                st.caption(
                    f"Zeitraum: {d0.date()} bis {d1.date()} | "
                    f"Skalierung: Startwert = {float(initial_investment):,.2f} USD"
                    + (" | Netto: 20% Steuer auf Gewinn (Endverkauf)" if is_netto else "")
                )

            mask = (df_e["date"].dt.date >= b_from) & (df_e["date"].dt.date <= b_to)
            e = df_e.loc[mask].copy()
            if e.empty:
                st.warning("Keine Daten im gewählten Zeitraum.")
            else:
                st.altair_chart(
                    alt.Chart(e)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Datum"),
                        y=alt.Y("adjClose:Q", title="adjClose"),
                        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("adjClose:Q", format=",.2f")],
                    )
                    .properties(height=360),
                    use_container_width=True,
                )
