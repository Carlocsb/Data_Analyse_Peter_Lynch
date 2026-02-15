# pages/Rendite_Dreieck_und_SP500.py
from __future__ import annotations

import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Pfad-Setup (Pages → src importierbar machen)
# ------------------------------------------------------------
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.funktionen import (  # noqa: E402
    get_es_connection,
    list_portfolios,
    load_portfolio,
)
from src.portfolio_simulation import (  # noqa: E402
    build_saved_buyhold_series_with_liquidation,
    get_buy_date_like_dynamic_for_portfolio,
    parse_portfolio_name,
    portfolio_doc_to_amount_weights,
    quarter_to_index,
    list_quarters_between,
    quarter_end_ts,
    geom_avg_quarter_return,
)

try:
    from src.portfolio_simulation import simulate_dynamic_cached  # noqa: E402

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

END_DATE_DEFAULT = pd.Timestamp("2025-12-31")

st.set_page_config(page_title="Rendite-Dreieck & Benchmarks", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png")
st.title("Rendite-Dreieck & Benchmarks")

es = get_es_connection()


def date_to_quarter_name(d: date) -> str:
    p = pd.Timestamp(d).to_period("Q")
    s = str(p)  # "YYYYQx"
    return s[:4] + "-Q" + s[-1]


# ============================================================
# ES Helpers
# ============================================================
def es_fetch_all_by_symbol(index: str, symbol: str, source_fields: List[str]) -> pd.DataFrame:
    """Pagination via search_after; erwartet sort=date asc."""
    page_size = 5000
    body: Dict[str, Any] = {
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

    rows: List[Dict[str, Any]] = []
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
    """Preis-Matrix (index=date, cols=symbols). Kein ffill (Delisting-Logik)."""
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
# Dreieck: Jahre (CAGR p.a.)
# ============================================================
def cagr(p0: float, p1: float, years: float) -> Optional[float]:
    if p0 <= 0 or p1 <= 0 or years <= 0:
        return None
    return (p1 / p0) ** (1.0 / years) - 1.0


def build_return_triangle(df: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    """df: columns date, adjClose"""
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


# ============================================================
# Dreieck: Jahre (CAGR p.a.) – NETTO (Steuer nur beim Endverkauf)
# ============================================================
def build_return_triangle_net(df: pd.DataFrame, year_from: int, year_to: int, tax_rate: float) -> pd.DataFrame:
    """Wie build_return_triangle, aber Netto: Steuer nur beim Verkauf auf Gewinn seit Kauf."""
    if df is None or df.empty:
        return pd.DataFrame()

    tax_r = max(0.0, min(1.0, float(tax_rate)))

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
            p1_brutto = price_by_year.get(sell)
            if p0 is None or p1_brutto is None:
                continue

            p0 = float(p0)
            p1_brutto = float(p1_brutto)
            years = float(sell - buy)
            if p0 <= 0 or p1_brutto <= 0 or years <= 0:
                continue

            gain = max(0.0, p1_brutto - p0)
            tax_paid = gain * tax_r
            p1_netto = p1_brutto - tax_paid

            v = cagr(p0, p1_netto, years)
            if v is None:
                continue

            rows.append(
                {
                    "buy_year": buy,
                    "sell_year": sell,
                    "cagr": float(v),  # NETTO-CAGR
                    "tax_paid": float(tax_paid),
                }
            )

    return pd.DataFrame(rows)


def compute_tri_vmin_vmax(tri: pd.DataFrame, value_col: str = "value") -> tuple[float, float]:
    v = pd.to_numeric(tri[value_col], errors="coerce").dropna()
    if v.empty:
        return -0.10, 0.10

    lo = float(v.quantile(0.05))
    hi = float(v.quantile(0.95))
    m = max(abs(lo), abs(hi))
    m = max(m, 0.10)
    return -m, m


def make_tri_color_scale(vmin: float, vmax: float, green_from: float) -> alt.Scale:
    # exakt wie Saved-Portfolio-Dreieck: Mitte ist "green_from"
    return alt.Scale(
        scheme="redyellowgreen",
        domain=[float(vmin), float(green_from), float(vmax)],
        clamp=True,
    )


# ============================================================
# Dreieck: Quartale (Ø Quartalsrendite geom.) – Steps (n_q)
# ============================================================
def build_quarter_triangle_steps(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    df_like: columns quarter, end_value
    value = (V_sell/V_buy)^(1/n_q) - 1
    """
    if df_like is None or df_like.empty:
        return pd.DataFrame()

    df = df_like.copy()
    df["quarter"] = df["quarter"].astype(str)
    df["end_value"] = pd.to_numeric(df["end_value"], errors="coerce")
    df = df.dropna(subset=["quarter", "end_value"])
    df = df[df["end_value"] > 0].copy()

    df["q_index"] = df["quarter"].map(quarter_to_index)
    df = df.dropna(subset=["q_index"]).copy()
    df["q_index"] = df["q_index"].astype(int)
    df = df.sort_values("q_index").reset_index(drop=True)

    q = df["quarter"].tolist()
    qi = df["q_index"].to_numpy()
    nav = df["end_value"].astype(float).to_numpy()

    rows = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            n_q = int(qi[j] - qi[i])
            if n_q <= 0:
                continue
            v0, v1 = nav[i], nav[j]
            value = geom_avg_quarter_return(v0, v1, n_q)
            rows.append({"buy_q": q[i], "sell_q": q[j], "value": float(value), "n_q": n_q})

    return pd.DataFrame(rows)


# ============================================================
# Dreieck: Quartale (Ø Quartalsrendite geom.) – NETTO (Steuer nur beim Endverkauf)
# ============================================================
def build_quarter_triangle_steps_net(df_like: pd.DataFrame, tax_rate: float) -> pd.DataFrame:
    """
    Wie build_quarter_triangle_steps, aber Netto:
    Steuer nur beim Verkauf auf Gewinn seit Kauf.
    value = (V_sell_netto/V_buy)^(1/n_q) - 1
    """
    if df_like is None or df_like.empty:
        return pd.DataFrame()

    tax_r = max(0.0, min(1.0, float(tax_rate)))

    df = df_like.copy()
    df["quarter"] = df["quarter"].astype(str)
    df["end_value"] = pd.to_numeric(df["end_value"], errors="coerce")
    df = df.dropna(subset=["quarter", "end_value"])
    df = df[df["end_value"] > 0].copy()

    df["q_index"] = df["quarter"].map(quarter_to_index)
    df = df.dropna(subset=["q_index"]).copy()
    df["q_index"] = df["q_index"].astype(int)
    df = df.sort_values("q_index").reset_index(drop=True)

    q = df["quarter"].tolist()
    qi = df["q_index"].to_numpy()
    nav = df["end_value"].astype(float).to_numpy()

    rows = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            n_q = int(qi[j] - qi[i])
            if n_q <= 0:
                continue

            v0 = float(nav[i])
            v1_brutto = float(nav[j])

            gain = max(0.0, v1_brutto - v0)
            tax_paid = gain * tax_r
            v1_netto = v1_brutto - tax_paid

            value = geom_avg_quarter_return(v0, v1_netto, n_q)
            if pd.isna(value):
                continue

            rows.append(
                {
                    "buy_q": q[i],
                    "sell_q": q[j],
                    "value": float(value),  # NETTO
                    "n_q": n_q,
                    "tax_paid": float(tax_paid),
                }
            )

    return pd.DataFrame(rows)


def plot_quarter_triangle_steps(
    tri: pd.DataFrame,
    title: str = "Ø Quartalsrendite (geom.)",
    green_from: float = 0.02,
) -> alt.Chart:
    def _key(qname: str) -> int:
        idx = quarter_to_index(qname)
        return idx if idx is not None else 10**9

    sell_order = sorted(tri["sell_q"].unique(), key=_key, reverse=True)
    buy_order = sorted(tri["buy_q"].unique(), key=_key, reverse=True)

    vmin, vmax = compute_tri_vmin_vmax(tri, value_col="value")
    color_scale = make_tri_color_scale(vmin, vmax, green_from)

    heat = (
        alt.Chart(tri)
        .mark_rect()
        .encode(
            x=alt.X("sell_q:O", title="Verkauf (Quartal)", sort=sell_order),
            y=alt.Y("buy_q:O", title="Kauf (Quartal)", sort=buy_order),
            color=alt.Color("value:Q", title=title, scale=color_scale),
            tooltip=[
                alt.Tooltip("buy_q:O", title="Kauf"),
                alt.Tooltip("sell_q:O", title="Verkauf"),
                alt.Tooltip("n_q:Q", title="Quartale (n)"),
                alt.Tooltip("value:Q", title=title, format=".1%"),
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
            text=alt.Text("value:Q", format=".1%"),
        )
    )
    return heat + text


# ============================================================
# EOQ helper
# ============================================================
def build_quarter_eoq_series_from_daily(df_daily: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    df_daily: columns ['date', value_col]
    returns: quarter, ld(=EOQ date), end_value
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    x = df_daily.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna(subset=["date", value_col]).sort_values("date")

    x["year"] = x["date"].dt.year
    x["q"] = x["date"].dt.quarter
    x["quarter"] = x["year"].astype(str) + "-Q" + x["q"].astype(str)

    eoq = x.groupby(["year", "q"], as_index=False).tail(1)[["quarter", "date", value_col]]
    eoq = eoq.rename(columns={"date": "ld", value_col: "end_value"})
    return eoq.sort_values("ld").reset_index(drop=True)


# ============================================================
# Benchmarks KPI
# ============================================================
def compute_benchmark_kpis(
    df_prices: pd.DataFrame,
    start_date: date,
    end_date: date,
    initial_capital: float,
) -> Dict[str, Any]:
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
    cagr_val = (
        (end_value / float(initial_capital)) ** (1.0 / years) - 1.0 if years and years > 0 else float("nan")
    )

    dd = (series / series.cummax()) - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    return {
        "status": "ok",
        "start_dt": d0,
        "end_dt": d1,
        "end_value": end_value,
        "total_return": total_return,
        "cagr": cagr_val,
        "max_dd": max_dd,
        "scale": scale,
    }


# ============================================================
# Saved Portfolio – NAV (Buy&Hold)
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def compute_saved_portfolio_nav(
    pid: str, base_capital: float, end_date: pd.Timestamp
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    doc = load_portfolio(es, pid)
    if not doc:
        return pd.DataFrame(), {"status": "missing_doc", "name": ""}

    weights = portfolio_doc_to_amount_weights(doc)
    if not weights:
        return pd.DataFrame(), {"status": "no_weights", "name": doc.get("name", "")}

    buy_dt = get_buy_date_like_dynamic_for_portfolio(es, STOCKS_INDEX, doc)
    if buy_dt is None:
        return pd.DataFrame(), {"status": "no_buy_dt", "name": doc.get("name", "")}

    syms = sorted(weights.keys())
    prices_mat = load_prices_matrix(syms)
    if prices_mat.empty:
        return pd.DataFrame(), {"status": "no_prices", "name": doc.get("name", "")}

    df_nav, _df_trades = build_saved_buyhold_series_with_liquidation(
        prices=prices_mat,
        weights=weights,
        buy_date=buy_dt,
        initial_capital=float(base_capital),
        end_date=end_date,
    )
    if df_nav is None or df_nav.empty:
        return pd.DataFrame(), {"status": "nav_empty", "name": doc.get("name", "")}

    df_nav = df_nav.copy()
    df_nav["date"] = pd.to_datetime(df_nav["date"], errors="coerce")
    df_nav["nav"] = pd.to_numeric(df_nav["nav"], errors="coerce")
    df_nav = df_nav.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)

    meta = {
        "status": "ok",
        "name": doc.get("name", ""),
        "buy_dt": pd.to_datetime(df_nav["date"].iloc[0]),
        "end_dt": pd.to_datetime(df_nav["date"].iloc[-1]),
    }
    return df_nav, meta


# ============================================================
# Saved Portfolios Triangle (alle Portfolios) – Netto-Steuer beim Verkauf
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def build_saved_portfolios_triangle(
    base_capital: float,
    q_start: str,
    q_end: str,
    end_date_cutoff: pd.Timestamp,
    tax_rate: float,  # wichtig: damit Cache + Dreieck auf Slider reagiert
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ports = list_portfolios(es) or []
    ports_q: List[Dict[str, Any]] = []
    for p in ports:
        name = p.get("name", "")
        pid = p.get("id")
        if pid and parse_portfolio_name(name) is not None:
            ports_q.append({"id": pid, "name": name})

    sell_quarters = list_quarters_between(q_start, q_end)
    if not sell_quarters:
        return pd.DataFrame(), pd.DataFrame([{"name": "", "id": "", "status": "bad_quarter_range"}])

    end_dt_qend = quarter_end_ts(q_end)
    if end_dt_qend is None:
        return pd.DataFrame(), pd.DataFrame([{"name": "", "id": "", "status": "bad_q_end"}])

    end_dt = min(pd.Timestamp(end_dt_qend), pd.Timestamp(end_date_cutoff))

    nav_cache: Dict[str, pd.DataFrame] = {}
    buy_q_by_pid: Dict[str, str] = {}
    status_rows: List[Dict[str, Any]] = []

    # NAV je Portfolio 1x rechnen
    for p in ports_q:
        pid = p["id"]
        df_nav, meta = compute_saved_portfolio_nav(pid, base_capital=base_capital, end_date=end_dt)
        status = meta.get("status", "unknown")
        status_rows.append({"name": p["name"], "id": pid, "status": status})
        if status == "ok" and df_nav is not None and not df_nav.empty:
            nav_cache[pid] = df_nav
            buy_q_by_pid[pid] = p["name"]

    # Steuer-Rate clampen
    tax_r = max(0.0, min(1.0, float(tax_rate)))

    # Dreiecks-Zellen bauen:
    # Steuer wird NUR beim Verkauf (Endverkauf) auf Gewinn seit Kauf fällig.
    rows: List[Dict[str, Any]] = []
    for pid, df_nav in nav_cache.items():
        buy_q = buy_q_by_pid.get(pid)
        if not buy_q:
            continue

        buy_idx = quarter_to_index(buy_q)
        if buy_idx is None:
            continue

        v0 = float(df_nav["nav"].iloc[0])

        for sell_q in sell_quarters:
            sell_idx = quarter_to_index(sell_q)
            if sell_idx is None or sell_idx <= buy_idx:
                continue

            n_q = int(sell_idx - buy_idx)
            sell_end = quarter_end_ts(sell_q)
            if sell_end is None:
                continue

            x = df_nav[df_nav["date"] <= pd.Timestamp(sell_end)]
            if x.empty:
                continue

            v1_brutto = float(x["nav"].iloc[-1])

            gain = max(0.0, v1_brutto - v0)
            tax_paid = gain * tax_r
            v1_netto = v1_brutto - tax_paid

            val = geom_avg_quarter_return(v0, v1_netto, n_q)
            if pd.isna(val):
                continue

            rows.append({"buy_q": buy_q, "sell_q": sell_q, "n_q": n_q, "value": float(val)})

    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def plot_saved_triangle(tri: pd.DataFrame, green_from: float = 0.0) -> alt.Chart:
    def _key(qname: str) -> int:
        idx = quarter_to_index(qname)
        return idx if idx is not None else 10**9

    sell_order = sorted(tri["sell_q"].unique(), key=_key, reverse=True)
    buy_order = sorted(tri["buy_q"].unique(), key=_key, reverse=True)

    v = pd.to_numeric(tri["value"], errors="coerce").dropna()
    if v.empty:
        vmin, vmax = -0.10, 0.10
    else:
        lo = float(v.quantile(0.05))
        hi = float(v.quantile(0.95))
        m = max(abs(lo), abs(hi))
        m = max(m, 0.10)
        vmin, vmax = -m, m

    color_scale = alt.Scale(scheme="redyellowgreen", domain=[vmin, float(green_from), vmax], clamp=True)

    heat = (
        alt.Chart(tri)
        .mark_rect()
        .encode(
            x=alt.X("sell_q:O", title="Verkauf (Quartal)", sort=sell_order),
            y=alt.Y("buy_q:O", title="Kaufquartal / Portfolio", sort=buy_order),
            color=alt.Color("value:Q", title="Ø Quartalsrendite (geom.)", scale=color_scale),
            tooltip=[
                alt.Tooltip("buy_q:O", title="Kauf"),
                alt.Tooltip("sell_q:O", title="Verkauf"),
                alt.Tooltip("n_q:Q", title="Quartale (n)"),
                alt.Tooltip("value:Q", title="Ø Quartalsrendite (geom.)", format=".2%"),
            ],
        )
        .properties(height=650)
    )

    text = (
        alt.Chart(tri)
        .mark_text(baseline="middle", fontSize=9)
        .encode(
            x=alt.X("sell_q:O", sort=sell_order),
            y=alt.Y("buy_q:O", sort=buy_order),
            text=alt.Text("value:Q", format=".1%"),
        )
    )
    return heat + text


# ============================================================
# Benchmark: Rendite-Dreieck (für Benchmark im Zeitraum) – NETTO
# ============================================================
def render_benchmark_triangle(
    df_prices: pd.DataFrame,
    triangle_view: str,
    b_from: date,
    b_to: date,
    year_from: int,
    year_to: int,
    green_from: float,
    tax_rate: float,
) -> None:
    if df_prices is None or df_prices.empty:
        st.warning("Keine Daten für Rendite-Dreieck gefunden.")
        return

    x = df_prices.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["adjClose"] = pd.to_numeric(x["adjClose"], errors="coerce")
    x = x.dropna(subset=["date", "adjClose"]).sort_values("date")

    mask = (x["date"].dt.date >= b_from) & (x["date"].dt.date <= b_to)
    x = x.loc[mask].copy()

    if x.empty or len(x) < 2:
        st.warning("Zu wenig Daten im Zeitraum für Rendite-Dreieck.")
        return

    if triangle_view.startswith("Quartale"):
        eoq = build_quarter_eoq_series_from_daily(x[["date", "adjClose"]], "adjClose")
        if eoq.empty or len(eoq) < 2:
            st.warning("Quartals-Dreieck konnte nicht berechnet werden.")
            return

        tri = build_quarter_triangle_steps_net(eoq[["quarter", "end_value"]].copy(), tax_rate=float(tax_rate))
        if tri.empty:
            st.warning("Quartals-Dreieck konnte nicht berechnet werden.")
            return

        st.altair_chart(
            plot_quarter_triangle_steps(tri, title="Ø Quartalsrendite (geom., netto)", green_from=green_from),
            use_container_width=True,
        )
        return

    min_y = int(x["date"].dt.year.min())
    max_y = int(x["date"].dt.year.max())
    y0 = max(int(year_from), min_y)
    y1 = min(int(year_to), max_y)

    if y1 - y0 < 2:
        st.warning(f"Zu wenig Jahresdaten für {y0}–{y1}.")
        return

    tri = build_return_triangle_net(x[["date", "adjClose"]].copy(), y0, y1, tax_rate=float(tax_rate))
    if tri.empty:
        st.warning("Jahres-Dreieck konnte nicht berechnet werden.")
        return

    v = pd.to_numeric(tri["cagr"], errors="coerce").dropna()
    if v.empty:
        vmin, vmax = -0.10, 0.10
    else:
        lo = float(v.quantile(0.05))
        hi = float(v.quantile(0.95))
        m = max(abs(lo), abs(hi))
        m = max(m, 0.10)
        vmin, vmax = -m, m

    color_scale = alt.Scale(scheme="redyellowgreen", domain=[vmin, float(green_from), vmax], clamp=True)

    heat = (
        alt.Chart(tri)
        .mark_rect()
        .encode(
            x=alt.X("sell_year:O", title="Verkauf (Jahr)", sort="descending"),
            y=alt.Y("buy_year:O", title="Kauf (Jahr)", sort="descending"),
            color=alt.Color("cagr:Q", title="CAGR p.a. (netto)", scale=color_scale),
            tooltip=[
                alt.Tooltip("buy_year:O", title="Kauf"),
                alt.Tooltip("sell_year:O", title="Verkauf"),
                alt.Tooltip("cagr:Q", title="CAGR p.a. (netto)", format=".2%"),
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


def render_benchmark_block(
    df_prices: pd.DataFrame,
    label: str,
    initial_investment: float,
    tax_rate: float,
    b_from: date,
    b_to: date,
    triangle_view: str,
    year_from: int,
    year_to: int,
    green_from: float,
) -> None:
    if df_prices is None or df_prices.empty:
        st.warning("Keine Daten gefunden.")
        return

    kpi = compute_benchmark_kpis(df_prices, b_from, b_to, float(initial_investment))
    if kpi.get("status") != "ok":
        st.warning(f"Benchmark KPI nicht verfügbar (Status: {kpi.get('status')}).")
        return

    v0 = float(initial_investment)
    end_brutto = float(kpi["end_value"])

    gain = max(0.0, end_brutto - v0)
    tax_r = max(0.0, min(1.0, float(tax_rate)))
    tax_paid = gain * tax_r
    end_netto = end_brutto - tax_paid

    start_q = date_to_quarter_name(pd.to_datetime(kpi["start_dt"]).date())
    end_q = date_to_quarter_name(pd.to_datetime(kpi["end_dt"]).date())
    i0 = quarter_to_index(start_q) or 0
    i1 = quarter_to_index(end_q) or 0
    n_q = max(1, int(i1 - i0)) if i1 > i0 else 1

    total_return_netto = (end_netto / v0) - 1.0 if v0 > 0 else float("nan")
    avg_q_netto = geom_avg_quarter_return(v0, end_netto, n_q)

    tab1, tab2 = st.tabs(["KPIs & Rendite-Dreieck", "Diagramm"])

    # -----------------------------
    # Tab 1: KPIs + Rendite-Dreieck
    # -----------------------------
    with tab1:
        kL, kR = st.columns(2)
        with kL:
            st.metric("Startbetrag", f"{v0:,.2f} USD")
            st.metric("Gesamtrendite (netto)", f"{total_return_netto*100:.1f}%")
            st.metric("Ø Quartalsrendite (geom., netto)", f"{avg_q_netto*100:.1f}%")
        with kR:
            st.metric("Endbetrag (brutto)", f"{end_brutto:,.2f} USD")
            st.metric("Endbetrag (netto)", f"{end_netto:,.2f} USD")
            st.metric("Steuer gesamt", f"{tax_paid:,.2f} USD")

        st.caption(
            f"Steuer wird NUR beim Endverkauf fällig: max(0, Endbetrag − Startbetrag) × Steuersatz ({tax_r*100:.0f}%)."
        )
        st.caption(f"Zeitraum: {start_q} → {end_q}  |  (Fenster: {b_from} → {b_to})")
        st.caption("Benchmark-Dreieck ist NETTO: pro Zelle Steuer nur beim Endverkauf auf Gewinn seit Kauf.")

        render_benchmark_triangle(
            df_prices=df_prices,
            triangle_view=triangle_view,
            b_from=b_from,
            b_to=b_to,
            year_from=int(year_from),
            year_to=int(year_to),
            green_from=float(green_from),
            tax_rate=float(tax_rate),
        )

    # -----------------------------
    # Tab 2: Diagramm (Linie)
    # -----------------------------
    with tab2:
        x = df_prices.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x["adjClose"] = pd.to_numeric(x["adjClose"], errors="coerce")
        x = x.dropna(subset=["date", "adjClose"]).sort_values("date")

        mask = (x["date"].dt.date >= b_from) & (x["date"].dt.date <= b_to)
        x = x.loc[mask].copy()

        if x.empty:
            st.info("Keine Daten im gewählten Zeitraum.")
        else:
            p0 = float(x["adjClose"].iloc[0])
            scale = (v0 / p0) if p0 > 0 else 1.0
            x["value_usd"] = x["adjClose"].astype(float) * float(scale)

            st.altair_chart(
                alt.Chart(x)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Datum"),
                    y=alt.Y("value_usd:Q", title=f"{label} (USD, skaliert)"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Datum"),
                        alt.Tooltip("value_usd:Q", title="Wert", format=",.2f"),
                    ],
                )
                .properties(height=320),
                use_container_width=True,
            )


# ============================================================
# Sidebar UI
# ============================================================
with st.sidebar:
    st.header("Einstellungen")

    mode = st.radio(
        "Quelle",
        ["Einzeltitel", "Gespeichertes Portfolio (Dreieck aller Portfolios)", "Dynamisch (Quartals-Umstieg)"],
        index=0,
    )

    initial_investment = st.number_input("Startkapital (USD)", min_value=0.0, value=1000.0, step=100.0)

    st.divider()
    tax_rate = st.number_input(
        "Steuersatz",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        key="tax_rate",
    )

    st.divider()
    show_bench = st.checkbox("Benchmarks anzeigen (GSPC & SPXEW)", value=True)
    if show_bench:
        with st.expander("Benchmark Zeitraum", expanded=False):
            b_from = st.date_input("Von Datum", value=date(2010, 1, 1), key="bench_from")
            b_to = st.date_input("Bis Datum", value=date.today(), key="bench_to")
    else:
        b_from = date(2010, 1, 1)
        b_to = date.today()

    st.divider()
    triangle_view = st.radio(
        "Dreieck",
        ["Jahre (CAGR p.a.)", "Quartale (Ø Quartalsrendite geom.)"],
        index=1,
    )

    st.divider()
    green_from = st.slider(
        "Ab welcher Ø-Quartalsrendite soll es Richtung Grün gehen?",
        min_value=-0.05,
        max_value=0.20,
        value=0.02,
        step=0.005,
        format="%.1f%%",
        key="green_from",
    )

    year_from = 2010
    year_to = 2025
    if triangle_view.startswith("Jahre"):
        year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
        year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)

    # Einzeltitel
    symbol: Optional[str] = None
    if mode == "Einzeltitel":
        symbols = get_symbols_from_prices()
        if symbols:
            default_sym = "AAPL" if "AAPL" in symbols else symbols[0]
            symbol = st.selectbox("Aktie", symbols, index=symbols.index(default_sym))
        else:
            st.warning(f"Keine Symbole im Index '{PRICE_INDEX}' gefunden.")

    # Gespeichertes Portfolio (Dreieck Range + Daten-Cutoff)
    q_start = "2010-Q1"
    q_end = "2025-Q4"
    end_cutoff = END_DATE_DEFAULT

    if mode.startswith("Gespeichertes Portfolio"):
        st.divider()
        st.caption("Dreieck Zeitraum (Quartale)")
        q_start = st.text_input("Start-Quartal (YYYY-Qx)", value="2010-Q1")
        q_end = st.text_input("End-Quartal (YYYY-Qx)", value="2025-Q4")
        end_cutoff = pd.to_datetime(st.date_input("Max. Enddatum (Daten-Cutoff)", value=END_DATE_DEFAULT.date()))


# ============================================================
# Main
# ============================================================
st.subheader("Auswertung")

# ============================================================
# A) Gespeichertes Portfolio: Dreieck + Strategie-Zeitraum
# ============================================================
if mode.startswith("Gespeichertes Portfolio"):
    st.subheader("Rendite-Dreieck (Saved Portfolios, Buy-&-Hold, Netto – Steuer beim Verkauf)")
    st.caption(
        "Jede Zeile ist ein Portfolio (Kaufquartal = Portfolio-Name). "
        "Spalten sind Verkaufsquartale. "
        "Zellen zeigen Ø Quartalsrendite (geom.) NACH Steuer (Steuer nur beim Endverkauf)."
    )

    tri, status_df = build_saved_portfolios_triangle(
        base_capital=1000.0,  # nur Skalierung; Renditen identisch
        q_start=q_start,
        q_end=q_end,
        end_date_cutoff=end_cutoff,
        tax_rate=float(tax_rate),  # WICHTIG: Dreieck reagiert auf Steuersatz
    )

    if tri.empty:
        st.warning("Dreieck konnte nicht gebaut werden. Prüfe Quartalsrange und Daten.")
        with st.expander("Debug: Portfolio-Status", expanded=True):
            st.dataframe(status_df, use_container_width=True, hide_index=True)
        st.stop()

    st.altair_chart(plot_saved_triangle(tri, green_from=float(green_from)), use_container_width=True)

    # --- Strategie-Zeitraum unter dem Dreieck ---
    st.subheader("Strategie-Zeitraum (Saved Portfolio)")
    st.caption("Start = Kaufquartal/Portfolio-Name (fix). Du wählst nur das Verkauf-Quartal.")

    ports_all = list_portfolios(es) or []
    ports_q: List[Dict[str, Any]] = []
    for p in ports_all:
        name = p.get("name", "")
        pid = p.get("id")
        if pid and parse_portfolio_name(name) is not None:
            ports_q.append({"id": pid, "name": name})

    if not ports_q:
        st.warning("Keine gültigen Saved-Portfolios (mit Quartalsname) gefunden.")
        st.stop()

    port_names = [p["name"] for p in ports_q]
    chosen_name = st.selectbox("Portfolio (Start = Kaufquartal)", port_names, index=0)
    chosen_pid = next(p["id"] for p in ports_q if p["name"] == chosen_name)

    buy_q = chosen_name
    buy_idx = quarter_to_index(buy_q)
    if buy_idx is None:
        st.warning("Portfolio-Name ist kein gültiges Quartal (YYYY-Qx).")
        st.stop()

    end_dt_qend = quarter_end_ts(q_end)
    if end_dt_qend is None:
        st.warning("End-Quartal (q_end) ist ungültig.")
        st.stop()
    end_dt = min(pd.Timestamp(end_dt_qend), pd.Timestamp(end_cutoff))

    df_nav, meta = compute_saved_portfolio_nav(chosen_pid, base_capital=1000.0, end_date=end_dt)
    if meta.get("status") != "ok" or df_nav.empty:
        st.warning(f"Keine NAV-Daten für dieses Portfolio. Status: {meta.get('status')}")
        st.stop()

    sell_quarters_all = list_quarters_between(buy_q, q_end)
    sell_quarters = [q for q in sell_quarters_all if (quarter_to_index(q) or -1) > buy_idx]
    sell_quarters = [
        q for q in sell_quarters if quarter_end_ts(q) is not None and pd.Timestamp(quarter_end_ts(q)) <= end_dt
    ]

    if not sell_quarters:
        st.info("Für dieses Portfolio gibt es innerhalb des Cutoffs kein Verkauf-Quartal nach dem Kaufquartal.")
        st.stop()

    sell_q = st.selectbox("Verkauf-Quartal", options=sell_quarters, index=len(sell_quarters) - 1)

    sell_idx = quarter_to_index(sell_q)
    if sell_idx is None or sell_idx <= buy_idx:
        st.warning("Verkauf-Quartal muss nach dem Start-Quartal liegen.")
        st.stop()

    n_q = int(sell_idx - buy_idx)
    v0_real = float(df_nav["nav"].iloc[0])

    sell_end = quarter_end_ts(sell_q)
    x = df_nav[df_nav["date"] <= pd.Timestamp(sell_end)]
    if x.empty:
        st.warning("Keine NAV-Daten bis zum gewählten Verkauf-Quartal.")
        st.stop()

    v1_real = float(x["nav"].iloc[-1])

    # Steuer NUR beim Verkauf (Endverkauf): auf Gewinn seit Kauf
    start_value = float(v0_real)
    end_brutto = float(v1_real)

    tax_r = max(0.0, min(1.0, float(tax_rate)))
    gain = max(0.0, end_brutto - start_value)
    tax_paid = gain * tax_r
    end_netto = end_brutto - tax_paid

    total_return_brutto = (end_brutto / start_value) - 1.0 if start_value > 0 else float("nan")
    total_return_netto = (end_netto / start_value) - 1.0 if start_value > 0 else float("nan")

    avg_q_brutto = geom_avg_quarter_return(start_value, end_brutto, n_q)
    avg_q_netto = geom_avg_quarter_return(start_value, end_netto, n_q)

    kL, kR = st.columns(2)
    with kL:
        st.metric("Startbetrag", f"{start_value:,.2f} USD")
        st.metric("Gesamtrendite (brutto)", f"{total_return_brutto*100:.1f}%")
        st.metric("Ø Quartalsrendite (geom., brutto)", f"{avg_q_brutto*100:.1f}%")
        st.metric("Gesamtrendite (netto)", f"{total_return_netto*100:.1f}%")
        st.metric("Ø Quartalsrendite (geom., netto)", f"{avg_q_netto*100:.1f}%")
    with kR:
        st.metric("Endbetrag (brutto)", f"{end_brutto:,.2f} USD")
        st.metric("Endbetrag (netto)", f"{end_netto:,.2f} USD")
        st.metric("Steuer gesamt", f"{tax_paid:,.2f} USD")
        st.metric("Quartale", f"{n_q}")

    st.caption(
        f"Steuer wird NUR beim Endverkauf fällig: max(0, Endbetrag − Startbetrag) × Steuersatz ({tax_r*100:.0f}%)."
    )
    st.caption(f"Zeitraum: {buy_q} → {sell_q}")

# ============================================================
# B) Einzeltitel
# ============================================================
elif mode == "Einzeltitel":
    if symbol is None:
        st.stop()

    df_base = load_prices(symbol)
    if df_base.empty:
        st.warning("Keine Preisdaten gefunden.")
        st.stop()

    st.subheader("Rendite-Dreieck")

    if triangle_view.startswith("Quartale"):
        eoq = build_quarter_eoq_series_from_daily(df_base[["date", "adjClose"]], "adjClose")
        if eoq.empty or len(eoq) < 2:
            st.warning("Quartals-Dreieck konnte nicht berechnet werden.")
        else:
            tri_q = build_quarter_triangle_steps(eoq[["quarter", "end_value"]].copy())
            st.altair_chart(plot_quarter_triangle_steps(tri_q, green_from=float(green_from)), use_container_width=True)

    else:
        df_base["date"] = pd.to_datetime(df_base["date"], errors="coerce")
        min_y = int(df_base["date"].dt.year.min())
        max_y = int(df_base["date"].dt.year.max())
        y0 = max(int(year_from), min_y)
        y1 = min(int(year_to), max_y)

        if y1 - y0 < 2:
            st.warning(f"Zu wenig Daten für {y0}–{y1}. Bitte Zeitraum erweitern.")
        else:
            tri_y = build_return_triangle(df_base, y0, y1)
            if tri_y.empty:
                st.warning("Rendite-Dreieck konnte nicht berechnet werden.")
            else:
                v = pd.to_numeric(tri_y["cagr"], errors="coerce").dropna()
                if v.empty:
                    vmin, vmax = -0.10, 0.10
                else:
                    lo = float(v.quantile(0.05))
                    hi = float(v.quantile(0.95))
                    m = max(abs(lo), abs(hi))
                    m = max(m, 0.10)
                    vmin, vmax = -m, m

                color_scale = alt.Scale(
                    scheme="redyellowgreen",
                    domain=[float(vmin), float(green_from), float(vmax)],
                    clamp=True,
                )

                heat = (
                    alt.Chart(tri_y)
                    .mark_rect()
                    .encode(
                        x=alt.X("sell_year:O", title="Verkauf (Jahr)", sort="descending"),
                        y=alt.Y("buy_year:O", title="Kauf (Jahr)", sort="descending"),
                        color=alt.Color("cagr:Q", title="CAGR p.a.", scale=color_scale),
                        tooltip=[
                            alt.Tooltip("buy_year:O", title="Kauf"),
                            alt.Tooltip("sell_year:O", title="Verkauf"),
                            alt.Tooltip("cagr:Q", title="CAGR p.a.", format=".2%"),
                        ],
                    )
                    .properties(height=520)
                )
                text = (
                    alt.Chart(tri_y)
                    .mark_text(baseline="middle", fontSize=11)
                    .encode(
                        x=alt.X("sell_year:O", sort="descending"),
                        y=alt.Y("buy_year:O", sort="descending"),
                        text=alt.Text("cagr:Q", format=".1%"),
                    )
                )
                st.altair_chart(heat + text, use_container_width=True)

# ============================================================
# C) Dynamisch: Dreieck -> Strategie-Zeitraum
# ============================================================
else:
    if not HAS_DYNAMIC:
        st.error("Dynamik ist aktuell nicht verfügbar (simulate_dynamic_cached fehlt).")
        st.stop()

    portfolios = list_portfolios(es) or []
    minimal = [{"id": p.get("id"), "name": p.get("name", "")} for p in portfolios if p.get("id")]

    if not minimal:
        st.warning("Keine gespeicherten Portfolios gefunden.")
        st.stop()

    df_dyn = simulate_dynamic_cached(
        _es_client=es,
        portfolio_minimal=minimal,
        initial_capital=float(initial_investment),
        tax_rate=float(tax_rate),
        year_from=int(year_from),
        year_to=int(year_to),
        prices_index=PRICE_INDEX,
        stocks_index=STOCKS_INDEX,
    )

    if df_dyn is None or df_dyn.empty:
        st.warning("Keine dynamischen Ergebnisse.")
        st.stop()

    df_dyn = df_dyn.copy()
    df_dyn["quarter"] = df_dyn["quarter"].astype(str)
    df_dyn["q_index"] = df_dyn["quarter"].map(quarter_to_index)
    df_dyn = df_dyn.dropna(subset=["q_index"]).copy()
    df_dyn["q_index"] = df_dyn["q_index"].astype(int)
    df_dyn = df_dyn.sort_values("q_index").reset_index(drop=True)

    st.subheader("Rendite-Dreieck (Dynamik)")

    if triangle_view.startswith("Quartale"):
        tri = build_quarter_triangle_steps(df_dyn[["quarter", "end_value"]].copy())
        if tri.empty:
            st.warning("Quartals-Dreieck konnte nicht berechnet werden.")
        else:
            st.altair_chart(plot_quarter_triangle_steps(tri, green_from=float(green_from)), use_container_width=True)
    else:
        df_base_year = df_dyn[["ld", "end_value"]].rename(columns={"ld": "date", "end_value": "adjClose"}).copy()
        df_base_year["date"] = pd.to_datetime(df_base_year["date"], errors="coerce")
        df_base_year["adjClose"] = pd.to_numeric(df_base_year["adjClose"], errors="coerce")
        df_base_year = df_base_year.dropna(subset=["date", "adjClose"])

        min_y = int(df_base_year["date"].dt.year.min())
        max_y = int(df_base_year["date"].dt.year.max())
        y0 = max(int(year_from), min_y)
        y1 = min(int(year_to), max_y)

        tri_year = build_return_triangle(df_base_year, y0, y1)
        if tri_year.empty:
            st.warning("Jahres-Dreieck konnte nicht berechnet werden.")
        else:
            v = pd.to_numeric(tri_year["cagr"], errors="coerce").dropna()
            if v.empty:
                vmin, vmax = -0.10, 0.10
            else:
                lo = float(v.quantile(0.05))
                hi = float(v.quantile(0.95))
                m = max(abs(lo), abs(hi))
                m = max(m, 0.10)
                vmin, vmax = -m, m

            color_scale = alt.Scale(
                scheme="redyellowgreen",
                domain=[float(vmin), float(green_from), float(vmax)],
                clamp=True,
            )

            heat = (
                alt.Chart(tri_year)
                .mark_rect()
                .encode(
                    x=alt.X("sell_year:O", title="Verkauf (Jahr)", sort="descending"),
                    y=alt.Y("buy_year:O", title="Kauf (Jahr)", sort="descending"),
                    color=alt.Color("cagr:Q", title="CAGR p.a.", scale=color_scale),
                    tooltip=[
                        alt.Tooltip("buy_year:O", title="Kauf"),
                        alt.Tooltip("sell_year:O", title="Verkauf"),
                        alt.Tooltip("cagr:Q", title="CAGR p.a.", format=".2%"),
                    ],
                )
                .properties(height=520)
            )
            text = (
                alt.Chart(tri_year)
                .mark_text(baseline="middle", fontSize=11)
                .encode(
                    x=alt.X("sell_year:O", sort="descending"),
                    y=alt.Y("buy_year:O", sort="descending"),
                    text=alt.Text("cagr:Q", format=".1%"),
                )
            )
            st.altair_chart(heat + text, use_container_width=True)

    # Strategie-Zeitraum
    st.subheader("Strategie-Zeitraum")

    quarter_opts = df_dyn["quarter"].tolist()
    s_col, e_col = st.columns(2)
    with s_col:
        start_q = st.selectbox("Start-Quartal", options=quarter_opts, index=0)
    with e_col:
        end_q = st.selectbox("End-Quartal", options=quarter_opts, index=len(quarter_opts) - 1)

    i0 = quarter_to_index(start_q)
    i1 = quarter_to_index(end_q)
    if i0 is None or i1 is None or i1 <= i0:
        st.warning("End-Quartal muss nach dem Start-Quartal liegen.")
        st.stop()

    df_path = df_dyn[(df_dyn["q_index"] >= i0) & (df_dyn["q_index"] <= i1)].copy()
    if df_path.empty:
        st.warning("Für diesen Zeitraum keine Daten gefunden.")
        st.stop()

    v0 = float(df_path.iloc[0]["end_value"])
    v1 = float(df_path.iloc[-1]["end_value"])
    tax_paid = float(pd.to_numeric(df_path.get("tax", 0.0), errors="coerce").fillna(0.0).sum())
    n_q = int(i1 - i0)

    total_return = (v1 / v0) - 1.0 if v0 > 0 else float("nan")
    avg_q = geom_avg_quarter_return(v0, v1, n_q)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Startbetrag (EOQ)", f"{v0:,.2f} USD")
        st.metric("Gesamtrendite", f"{total_return*100:.1f}%")
        st.metric("Ø Quartalsrendite (geom.)", f"{avg_q*100:.1f}%")
    with c2:
        st.metric("Endbetrag", f"{v1:,.2f} USD")
        st.metric("Steuer gesamt", f"{tax_paid:,.2f} USD")
        st.metric("Quartale", f"{n_q}")

    with st.expander("Vergleich: Start = 1.000 USD (gleicher Zeitraum)", expanded=False):
        base_compare = 1000.0
        scale = (base_compare / v0) if v0 > 0 else float("nan")

        v0_1000 = base_compare
        v1_1000 = v1 * scale
        tax_1000 = tax_paid * scale

        total_return_1000 = (v1_1000 / v0_1000) - 1.0 if v0_1000 > 0 else float("nan")
        avg_q_1000 = geom_avg_quarter_return(v0_1000, v1_1000, n_q)

        cL, cR = st.columns(2)
        with cL:
            st.metric("Startbetrag (EOQ)", f"{v0_1000:,.2f} USD")
            st.metric("Gesamtrendite", f"{total_return_1000*100:.1f}%")
            st.metric("Ø Quartalsrendite (geom.)", f"{avg_q_1000*100:.1f}%")
        with cR:
            st.metric("Endbetrag", f"{v1_1000:,.2f} USD")
            st.metric("Steuer gesamt", f"{tax_1000:,.2f} USD")
            st.metric("Quartale", f"{n_q}")

        st.caption("Renditen identisch – USD-Beträge sind so skaliert, dass der Start im gewählten Fenster 1.000 USD ist.")


# ============================================================
# Benchmarks (am Ende, für alle Modi)
# ============================================================
if show_bench:
    st.divider()
    st.subheader("Benchmarks")

    with st.expander("S&P 500 (GSPC)", expanded=False):
        render_benchmark_block(
            df_prices=load_gspc(),
            label="GSPC",
            initial_investment=float(initial_investment),
            tax_rate=float(tax_rate),
            b_from=b_from,
            b_to=b_to,
            triangle_view=str(triangle_view),
            year_from=int(year_from),
            year_to=int(year_to),
            green_from=float(green_from),
        )

    with st.expander("S&P 500 Equal Weight (SPXEW)", expanded=False):
        render_benchmark_block(
            df_prices=load_spxew(),
            label="SPXEW",
            initial_investment=float(initial_investment),
            tax_rate=float(tax_rate),
            b_from=b_from,
            b_to=b_to,
            triangle_view=str(triangle_view),
            year_from=int(year_from),
            year_to=int(year_to),
            green_from=float(green_from),
        )
