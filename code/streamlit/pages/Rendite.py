# pages/Rendite_Dreieck_und_SP500.py
from __future__ import annotations
import os
import sys
import re
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

from src.funktionen import (  # noqa: E402
    get_es_connection,
    list_portfolios,
    load_portfolio,
)

from src.portfolio_simulation import (  # noqa: E402
    parse_portfolio_name,
    portfolio_doc_to_amount_weights,
    get_buy_date_like_dynamic_for_portfolio,
    build_saved_buyhold_series_with_liquidation,
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

# ============================================================
# Quarter helpers
# ============================================================
_QRE = re.compile(r"^(?P<y>\d{4})-Q(?P<q>[1-4])$")


def quarter_to_index(qname: str) -> Optional[int]:
    m = _QRE.match(str(qname).strip())
    if not m:
        return None
    return int(m.group("y")) * 4 + (int(m.group("q")) - 1)


def index_to_quarter(qi: int) -> str:
    y = qi // 4
    q = (qi % 4) + 1
    return f"{y}-Q{q}"


def list_quarters_between(q_start: str, q_end: str) -> List[str]:
    i0 = quarter_to_index(q_start)
    i1 = quarter_to_index(q_end)
    if i0 is None or i1 is None or i1 < i0:
        return []
    return [index_to_quarter(i) for i in range(i0, i1 + 1)]


def quarter_end_ts(qname: str) -> Optional[pd.Timestamp]:
    m = _QRE.match(str(qname).strip())
    if not m:
        return None
    y = int(m.group("y"))
    q = int(m.group("q"))
    if q == 1:
        return pd.Timestamp(year=y, month=3, day=31)
    if q == 2:
        return pd.Timestamp(year=y, month=6, day=30)
    if q == 3:
        return pd.Timestamp(year=y, month=9, day=30)
    return pd.Timestamp(year=y, month=12, day=31)


def geom_avg_quarter_return(v0: float, v1: float, n_quarters: int) -> float:
    """(v1/v0)^(1/n) - 1"""
    if v0 <= 0 or v1 <= 0 or n_quarters <= 0:
        return float("nan")
    return (v1 / v0) ** (1.0 / float(n_quarters)) - 1.0


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
# Dreieck: Jahre (CAGR p.a.)
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


# ============================================================
# Dreieck: Quartale (Ø Quartalsrendite geom.) – Steps (n_q)
# ============================================================
def build_quarter_triangle_steps(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    df_like: quarter, end_value
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


def plot_quarter_triangle_steps(tri: pd.DataFrame, title: str = "Ø Quartalsrendite (geom.)") -> alt.Chart:
    def _key(qname: str) -> int:
        idx = quarter_to_index(qname)
        return idx if idx is not None else 10**9

    sell_order = sorted(tri["sell_q"].unique(), key=_key, reverse=True)
    buy_order = sorted(tri["buy_q"].unique(), key=_key, reverse=True)

    # --------- NEU: robuste Skalierung ---------
    v = pd.to_numeric(tri["value"], errors="coerce").dropna()
    if v.empty:
        vmin, vmax = -0.10, 0.10
    else:
        # Quantile, damit Ausreißer (z.B. +100%) nicht alles "plattdrücken"
        lo = float(v.quantile(0.05))
        hi = float(v.quantile(0.95))
        # symmetrisch um 0 (damit grün/rot fair verteilt ist)
        m = max(abs(lo), abs(hi))
        # Mindestbreite, sonst wird’s zu aggressiv
        m = max(m, 0.10)  # 10% pro Quartal als floor
        vmin, vmax = -m, m

    color_scale = alt.Scale(
        scheme="redyellowgreen",
        domain=[vmin, 0.0, vmax],  # 0 liegt genau in der Mitte
        clamp=True
    )
    # ------------------------------------------

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
                alt.Tooltip("value:Q", title=title, format=".2%"),
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
    df_daily: columns ['date', value_col] (date as datetime)
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
    cagr_val = (end_value / float(initial_capital)) ** (1.0 / years) - 1.0 if years and years > 0 else float("nan")

    dd = (series / series.cummax()) - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")

    eoq = build_quarter_eoq_series_from_daily(x[["date", "adjClose"]], "adjClose")
    if eoq.empty or len(eoq) < 2:
        best_q = worst_q = avg_q = float("nan")
    else:
        q_ret = eoq["end_value"].pct_change()
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


# ============================================================
# Saved Portfolio – NAV (Buy&Hold) + EOQ
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def compute_saved_portfolio_nav(pid: str, base_capital: float, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Liefert tägliche NAV-Serie (brutto) für ein Saved Portfolio als Buy-&-Hold.
    base_capital: nur Skalierung (Renditen unabhängig)
    """
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


def nav_to_eoq(df_nav: pd.DataFrame) -> pd.DataFrame:
    """df_nav: date, nav -> quarter, ld, end_value (EOQ nav)"""
    if df_nav is None or df_nav.empty:
        return pd.DataFrame()
    x = df_nav[["date", "nav"]].rename(columns={"nav": "end_value"}).copy()
    return build_quarter_eoq_series_from_daily(x, "end_value")


# ============================================================
# NEW: Saved Portfolios Triangle (no selection)
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def build_saved_portfolios_triangle(
    base_capital: float,
    q_start: str,
    q_end: str,
    end_date_cutoff: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build triangle where each row = one portfolio (buy_q = portfolio name),
    each col = sell quarter, value = geom avg quarterly return.
    Returns: (tri_df, status_df)
    """
    ports = list_portfolios(es)
    ports_q = []
    for p in (ports or []):
        name = p.get("name", "")
        if p.get("id") and parse_portfolio_name(name) is not None:
            ports_q.append({"id": p["id"], "name": name})

    sell_quarters = list_quarters_between(q_start, q_end)
    if not sell_quarters:
        return pd.DataFrame(), pd.DataFrame([{"name": "", "id": "", "status": "bad_quarter_range"}])

    end_dt = quarter_end_ts(q_end)
    if end_dt is None:
        return pd.DataFrame(), pd.DataFrame([{"name": "", "id": "", "status": "bad_q_end"}])

    # never exceed cutoff
    end_dt = min(pd.Timestamp(end_dt), pd.Timestamp(end_date_cutoff))

    nav_cache: Dict[str, pd.DataFrame] = {}
    buy_q_by_pid: Dict[str, str] = {}
    status_rows: List[Dict[str, Any]] = []

    for p in ports_q:
        pid = p["id"]
        df_nav, meta = compute_saved_portfolio_nav(pid, base_capital=base_capital, end_date=end_dt)
        status = meta.get("status", "unknown")
        status_rows.append({"name": p["name"], "id": pid, "status": status})

        if status != "ok" or df_nav.empty:
            continue

        nav_cache[pid] = df_nav
        buy_q_by_pid[pid] = p["name"]

    # build triangle rows
    rows: List[Dict[str, Any]] = []
    for pid, df_nav in nav_cache.items():
        buy_q = buy_q_by_pid.get(pid)
        if not buy_q:
            continue

        buy_idx = quarter_to_index(buy_q)
        if buy_idx is None:
            continue

        # start value is NAV at first nav date
        v0 = float(df_nav["nav"].iloc[0])

        for sell_q in sell_quarters:
            sell_idx = quarter_to_index(sell_q)
            if sell_idx is None or sell_idx <= buy_idx:
                continue

            n_q = int(sell_idx - buy_idx)
            sell_end = quarter_end_ts(sell_q)
            if sell_end is None:
                continue

            # value at or before sell_end
            x = df_nav[df_nav["date"] <= pd.Timestamp(sell_end)]
            if x.empty:
                continue
            v1 = float(x["nav"].iloc[-1])

            val = geom_avg_quarter_return(v0, v1, n_q)
            if pd.isna(val):
                continue

            rows.append({"portfolio": buy_q, "buy_q": buy_q, "sell_q": sell_q, "n_q": n_q, "value": float(val)})

    tri = pd.DataFrame(rows)
    status_df = pd.DataFrame(status_rows)
    return tri, status_df


def plot_saved_triangle(tri: pd.DataFrame, green_from: float = 0.0) -> alt.Chart:
    def _key(qname: str) -> int:
        idx = quarter_to_index(qname)
        return idx if idx is not None else 10**9

    sell_order = sorted(tri["sell_q"].unique(), key=_key, reverse=True)
    buy_order = sorted(tri["buy_q"].unique(), key=_key, reverse=True)

    # robuste Skalierung (Quantile) + einstellbarer "Neutralpunkt" green_from
    v = pd.to_numeric(tri["value"], errors="coerce").dropna()
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
        domain=[vmin, float(green_from), vmax],
        clamp=True,
    )

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
# Sidebar UI
# ============================================================
green_from = 0.05

with st.sidebar:
    st.header("Einstellungen")

    mode = st.radio(
        "Quelle",
        ["Einzeltitel", "Gespeichertes Portfolio (Dreieck aller Portfolios)", "Dynamisch (Quartals-Umstieg)"],
        index=0,
    )

    initial_investment = st.number_input("Startkapital (USD)", min_value=0.0, value=1000.0, step=100.0)

    # Saved triangle range (no selection)
    if mode.startswith("Gespeichertes Portfolio"):
        st.divider()
        st.caption("Dreieck Zeitraum (Quartale)")
        q_start = st.text_input("Start-Quartal (YYYY-Qx)", value="2010-Q1")
        q_end = st.text_input("End-Quartal (YYYY-Qx)", value="2025-Q4")
        end_cutoff = pd.to_datetime(st.date_input("Max. Enddatum (Daten-Cutoff)", value=END_DATE_DEFAULT.date()))
        st.divider()
        green_from = st.slider(
        "Ab welcher Ø-Quartalsrendite soll es Richtung Grün gehen?",
        min_value=-0.05, max_value=0.20, value=0.02, step=0.005,
        format="%.1f%%"
    )

    else:
        q_start = "2010-Q1"
        q_end = "2025-Q4"
        end_cutoff = END_DATE_DEFAULT

    show_bench = st.checkbox("Benchmarks anzeigen (GSPC & SPXEW)", value=True)
    if show_bench:
        with st.expander("Benchmark Zeitraum", expanded=False):
            b_from = st.date_input("Von Datum", value=date(2010, 1, 1))
            b_to = st.date_input("Bis Datum", value=date.today())
        bench_view = st.radio(
            "Benchmark Anzeige",
            ["Brutto", "Netto (20% Steuer auf Gewinn)"],
            index=0,
            horizontal=True,
        )
    else:
        b_from = date(2010, 1, 1)
        b_to = date.today()
        bench_view = "Brutto"

    # Einzeltitel
    symbol: Optional[str] = None
    triangle_view = "Quartale (Ø Quartalsrendite geom.)"
    year_from = 2010
    year_to = 2025

    # Dynamik
    dyn_tax_rate = 0.20

    if mode == "Einzeltitel":
        symbols = get_symbols_from_prices()
        if symbols:
            default_sym = "AAPL" if "AAPL" in symbols else symbols[0]
            symbol = st.selectbox("Aktie", symbols, index=symbols.index(default_sym))
        else:
            st.warning(f"Keine Symbole im Index '{PRICE_INDEX}' gefunden.")

        st.divider()
        triangle_view = st.radio(
            "Dreieck",
            ["Jahre (CAGR p.a.)", "Quartale (Ø Quartalsrendite geom.)"],
            index=1,
        )

        if triangle_view.startswith("Jahre"):
            st.divider()
            year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
            year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)

    elif mode.startswith("Gespeichertes Portfolio"):
        # nothing else; triangle is shown without selecting one portfolio
        pass

    else:
        if not HAS_DYNAMIC:
            st.warning("Dynamik nicht verfügbar (simulate_dynamic_cached fehlt).")

        st.divider()
        dyn_tax_rate = st.number_input(
            "Dynamik Steuersatz (pro Umschichtung)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )

        st.divider()
        triangle_view = st.radio(
            "Dreieck",
            ["Jahre (CAGR p.a.)", "Quartale (Ø Quartalsrendite geom.)"],
            index=1,
        )

        if triangle_view.startswith("Jahre"):
            st.divider()
            year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
            year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)


# ============================================================
# Main
# ============================================================
st.subheader("Auswertung")

# ============================================================
# A) Saved Portfolios Triangle (NO selection)
# ============================================================
if mode.startswith("Gespeichertes Portfolio"):
    st.subheader("Rendite-Dreieck (Saved Portfolios, Buy-&-Hold, Brutto)")
    st.caption(
        "Jede Zeile ist ein Portfolio (Kaufquartal = Portfolio-Name). "
        "Spalten sind Verkaufsquartale. "
        "Zellen zeigen Ø Quartalsrendite (geom.)."
    )

    tri, status_df = build_saved_portfolios_triangle(
        base_capital=1000.0,   # only for internal scaling; returns identical
        q_start=q_start,
        q_end=q_end,
        end_date_cutoff=end_cutoff,
    )

    if tri.empty:
        st.warning("Dreieck konnte nicht gebaut werden. Prüfe Quartalsrange und Daten.")
        with st.expander("Debug: Portfolio-Status", expanded=True):
            st.dataframe(status_df, use_container_width=True, hide_index=True)
        st.stop()

    st.altair_chart(plot_saved_triangle(tri, green_from=green_from), use_container_width=True)


    with st.expander("Debug: Portfolio-Status", expanded=False):
        st.dataframe(status_df.sort_values(["status", "name"]), use_container_width=True, hide_index=True)

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
            tri = build_quarter_triangle_steps(eoq[["quarter", "end_value"]].copy())
            st.altair_chart(plot_quarter_triangle_steps(tri, title="Ø Quartalsrendite (geom.)"), use_container_width=True)
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

# ============================================================
# C) Dynamisch
# ============================================================
else:
    if not HAS_DYNAMIC:
        st.error("Dynamik ist aktuell nicht verfügbar (simulate_dynamic_cached fehlt).")
        st.stop()

    portfolios = list_portfolios(es)
    if not portfolios:
        st.warning("Keine gespeicherten Portfolios gefunden.")
        st.stop()

    minimal = [{"id": p.get("id"), "name": p.get("name", "")} for p in portfolios if p.get("id")]

    df_dyn = simulate_dynamic_cached(
        _es_client=es,
        portfolio_minimal=minimal,
        initial_capital=float(initial_investment),
        tax_rate=float(dyn_tax_rate),
        year_from=int(year_from),
        year_to=int(year_to),
        prices_index=PRICE_INDEX,
        stocks_index=STOCKS_INDEX,
    )

    if df_dyn.empty:
        st.warning("Keine dynamischen Ergebnisse.")
        st.stop()

    st.subheader("Rendite-Dreieck (Dynamik)")

    if triangle_view.startswith("Quartale"):
        tri = build_quarter_triangle_steps(df_dyn[["quarter", "end_value"]].copy())
        if tri.empty:
            st.warning("Quartals-Dreieck konnte nicht berechnet werden.")
        else:
            st.altair_chart(plot_quarter_triangle_steps(tri, title="Ø Quartalsrendite (geom.)"), use_container_width=True)
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
            st.warning("Jahres-Dreieck konnte nicht berechnet werden.")
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

    # KPI (optional)
    st.subheader("Dynamische Strategie – Ergebnisse")
    endbetrag = float(df_dyn.iloc[-1]["end_value"])
    total_return = (endbetrag / float(initial_investment)) - 1.0 if initial_investment > 0 else float("nan")
    nav = df_dyn["end_value"].astype(float)
    max_dd = float(((nav / nav.cummax()) - 1.0).min()) if len(nav) else float("nan")
    sum_tax = float(df_dyn["tax"].sum())
    best_q = float(df_dyn["return_q"].max())
    worst_q = float(df_dyn["return_q"].min())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Endbetrag", f"{endbetrag:,.2f} USD")
    k2.metric("Gesamtrendite", f"{total_return*100:.1f}%")
    k3.metric("Max Drawdown", f"{max_dd*100:.1f}%")
    k4.metric("Summe Steuer", f"{sum_tax:,.2f} USD")

    k5, k6 = st.columns(2)
    k5.metric("Bestes Quartal", f"{best_q*100:.2f}%")
    k6.metric("Schlechtestes Quartal", f"{worst_q*100:.2f}%")


# ============================================================
# Benchmarks (optional, für alle Modi)
# ============================================================
if show_bench:
    st.divider()
    st.subheader("Benchmarks (Linien)")

    tax_rate_bench = 0.20
    is_netto = bench_view.startswith("Netto")

    with st.expander("S&P 500 (GSPC)", expanded=False):
        df_g = load_gspc()
        if df_g.empty:
            st.warning(f"Keine Daten für {GSPC_SYMBOL} im Index '{BENCH_INDEX}' gefunden.")
        else:
            kpi_g = compute_benchmark_kpis(df_g, b_from, b_to, float(initial_investment))
            if kpi_g.get("status") == "ok":
                end_brutto = float(kpi_g["end_value"])
                gain = end_brutto - float(initial_investment)
                bench_tax = max(0.0, gain) * tax_rate_bench if is_netto else 0.0
                end_display = end_brutto - bench_tax if is_netto else end_brutto
                total_return_display = (end_display / float(initial_investment)) - 1.0 if initial_investment > 0 else float("nan")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Endbetrag ({'netto' if is_netto else 'brutto'})", f"{end_display:,.2f} USD")
                k2.metric("Gesamtrendite", f"{total_return_display*100:.1f}%")
                k3.metric("CAGR p.a.", f"{float(kpi_g['cagr'])*100:.1f}%")
                k4.metric("Max Drawdown", f"{float(kpi_g['max_dd'])*100:.1f}%")

                st.caption("Netto = 20% Steuer auf Gewinn am Endverkauf." if is_netto else "Brutto (ohne Steuern).")

            mask = (df_g["date"].dt.date >= b_from) & (df_g["date"].dt.date <= b_to)
            g = df_g.loc[mask].copy()
            if not g.empty:
                st.altair_chart(
                    alt.Chart(g)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Datum"),
                        y=alt.Y("adjClose:Q", title="adjClose"),
                        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("adjClose:Q", format=",.2f")],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )

    with st.expander("S&P 500 Equal Weight (SPXEW)", expanded=False):
        df_e = load_spxew()
        if df_e.empty:
            st.warning(f"Keine Daten für {SPXEW_SYMBOL} im Index '{BENCH_INDEX}' gefunden.")
        else:
            kpi_e = compute_benchmark_kpis(df_e, b_from, b_to, float(initial_investment))
            if kpi_e.get("status") == "ok":
                end_brutto = float(kpi_e["end_value"])
                gain = end_brutto - float(initial_investment)
                bench_tax = max(0.0, gain) * tax_rate_bench if is_netto else 0.0
                end_display = end_brutto - bench_tax if is_netto else end_brutto
                total_return_display = (end_display / float(initial_investment)) - 1.0 if initial_investment > 0 else float("nan")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Endbetrag ({'netto' if is_netto else 'brutto'})", f"{end_display:,.2f} USD")
                k2.metric("Gesamtrendite", f"{total_return_display*100:.1f}%")
                k3.metric("CAGR p.a.", f"{float(kpi_e['cagr'])*100:.1f}%")
                k4.metric("Max Drawdown", f"{float(kpi_e['max_dd'])*100:.1f}%")

                st.caption("Netto = 20% Steuer auf Gewinn am Endverkauf." if is_netto else "Brutto (ohne Steuern).")

            mask = (df_e["date"].dt.date >= b_from) & (df_e["date"].dt.date <= b_to)
            e = df_e.loc[mask].copy()
            if not e.empty:
                st.altair_chart(
                    alt.Chart(e)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Datum"),
                        y=alt.Y("adjClose:Q", title="adjClose"),
                        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("adjClose:Q", format=",.2f")],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )
