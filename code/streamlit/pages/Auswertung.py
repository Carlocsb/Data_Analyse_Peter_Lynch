# pages/test_diagonalen_boxplots.py
from __future__ import annotations

import os
import sys
from datetime import date
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(page_title="Auswertung", layout="wide")
st.title("Auswertung")

# ------------------------------------------------------------
# Pfad-Setup
# ------------------------------------------------------------
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.funktionen import get_es_connection, list_portfolios, load_portfolio  # noqa: E402
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
MIN_NQ = 4

es = get_es_connection()


# ============================================================
# ES helpers
# ============================================================
def es_fetch_all_by_symbol(index: str, symbol: str, source_fields: List[str]) -> pd.DataFrame:
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


@st.cache_data(show_spinner=True, ttl=300)
def load_prices(symbol: str) -> pd.DataFrame:
    return es_fetch_all_by_symbol(PRICE_INDEX, symbol, ["symbol", "date", "adjClose"])


@st.cache_data(show_spinner=True, ttl=300)
def load_prices_matrix(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        d = load_prices(sym)
        if d.empty:
            continue
        frames.append(d[["date", "adjClose"]].rename(columns={"adjClose": sym}).set_index("date"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


@st.cache_data(show_spinner=True, ttl=300)
def load_gspc() -> pd.DataFrame:
    return es_fetch_all_by_symbol(BENCH_INDEX, GSPC_SYMBOL, ["symbol", "date", "adjClose"])


@st.cache_data(show_spinner=True, ttl=300)
def load_spxew() -> pd.DataFrame:
    return es_fetch_all_by_symbol(BENCH_INDEX, SPXEW_SYMBOL, ["symbol", "date", "adjClose"])


# ============================================================
# EOQ helper (Daily -> End of Quarter)
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
# Plot helpers
# ============================================================
def plot_counts_by_nq_range(d: pd.DataFrame, title: str, n_min: int, n_max: int, height: int = 160) -> alt.Chart:
    x = d.copy()
    x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
    x = x.dropna(subset=["n_q"]).copy()
    x["n_q"] = x["n_q"].astype(int)
    x = x[(x["n_q"] >= int(n_min)) & (x["n_q"] <= int(n_max))].copy()

    if x.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    counts = x.groupby("n_q").size().reset_index(name="count").sort_values("n_q")
    label_angle = 0 if (n_max - n_min) <= 12 else 90

    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("n_q:O", title="Haltedauer (Quartale)", axis=alt.Axis(labelAngle=label_angle)),
            y=alt.Y("count:Q", title="Anzahl Werte"),
            tooltip=[alt.Tooltip("n_q:O"), alt.Tooltip("count:Q")],
        )
        .properties(title=title, height=height)
    )


def plot_boxplots_by_nq_range(d: pd.DataFrame, title: str, n_min: int, n_max: int, height: int = 280) -> alt.Chart:
    x = d[(d["n_q"] >= int(n_min)) & (d["n_q"] <= int(n_max))].copy()
    if x.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    label_angle = 0 if (n_max - n_min) <= 12 else 90

    return (
        alt.Chart(x)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("n_q:O", title="Haltedauer (Quartale)", axis=alt.Axis(labelAngle=label_angle)),
            y=alt.Y("value:Q", title="Ø Quartalsrendite (geom.)", axis=alt.Axis(format=".1%")),
            tooltip=[
                alt.Tooltip("n_q:O", title="n_q"),
                alt.Tooltip("value:Q", title="Rendite", format=".2%"),
                alt.Tooltip("buy_q:O", title="Buy"),
                alt.Tooltip("sell_q:O", title="Sell"),
            ],
        )
        .properties(title=title, height=height)
    )


# ============================================================
# Median-Linien
# ============================================================
def median_by_nq(d: pd.DataFrame, label: str) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame(columns=["n_q", "median", "mode"])

    x = d.copy()
    x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna(subset=["n_q", "value"]).copy()
    x["n_q"] = x["n_q"].astype(int)

    med = (
        x.groupby("n_q")["value"]
        .median()
        .reset_index(name="median")
        .sort_values("n_q")
        .reset_index(drop=True)
    )
    med["mode"] = str(label)
    return med


def plot_median_lines(
    med: pd.DataFrame,
    title: str,
    n_min: int,
    n_max: int,
    height: int = 260,
) -> alt.Chart:
    if med is None or med.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    x = med.copy()
    x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
    x["median"] = pd.to_numeric(x["median"], errors="coerce")
    x = x.dropna(subset=["n_q", "median"]).copy()
    x["n_q"] = x["n_q"].astype(int)
    x = x[(x["n_q"] >= int(n_min)) & (x["n_q"] <= int(n_max))].copy()

    if x.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    label_angle = 0 if (n_max - n_min) <= 12 else 90

    return (
        alt.Chart(x)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_q:O", title="Haltedauer (Quartale)", axis=alt.Axis(labelAngle=label_angle)),
            y=alt.Y("median:Q", title="Median Ø Quartalsrendite (geom.)", axis=alt.Axis(format=".1%")),
            color=alt.Color("mode:N", title="Serie"),
            tooltip=[
                alt.Tooltip("mode:N", title="Serie"),
                alt.Tooltip("n_q:O", title="n_q"),
                alt.Tooltip("median:Q", title="Median", format=".2%"),
            ],
        )
        .properties(title=title, height=height)
    )


def plot_median_lines_multi(
    med_all: pd.DataFrame,
    title: str,
    n_min: int,
    n_max: int,
    height: int = 320,
) -> alt.Chart:
    if med_all is None or med_all.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    x = med_all.copy()
    x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
    x["median"] = pd.to_numeric(x["median"], errors="coerce")
    x = x.dropna(subset=["n_q", "median"]).copy()
    x["n_q"] = x["n_q"].astype(int)
    x = x[(x["n_q"] >= int(n_min)) & (x["n_q"] <= int(n_max))].copy()

    if x.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    label_angle = 0 if (n_max - n_min) <= 12 else 90

    return (
        alt.Chart(x)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_q:O", title="Haltedauer (Quartale)", axis=alt.Axis(labelAngle=label_angle)),
            y=alt.Y("median:Q", title="Median Ø Quartalsrendite (geom.)", axis=alt.Axis(format=".1%")),
            color=alt.Color("mode:N", title="Serie"),
            tooltip=[
                alt.Tooltip("mode:N", title="Serie"),
                alt.Tooltip("n_q:O", title="n_q"),
                alt.Tooltip("median:Q", title="Median", format=".2%"),
            ],
        )
        .properties(title=title, height=height)
    )


# ============================================================
# Vergleich Saved vs Dynamik (nebeneinander)
# ============================================================
def stack_saved_dyn(d_saved: pd.DataFrame, d_dyn: pd.DataFrame) -> pd.DataFrame:
    a = d_saved.copy()
    a["mode"] = "Saved"
    b = d_dyn.copy()
    b["mode"] = "Dynamik"

    cols = ["n_q", "value", "mode", "buy_q", "sell_q"]
    out = pd.concat([a[cols], b[cols]], ignore_index=True)

    out["n_q"] = pd.to_numeric(out["n_q"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["n_q", "value"]).copy()
    out["n_q"] = out["n_q"].astype(int)
    return out


def plot_boxplots_saved_vs_dyn(
    d_both: pd.DataFrame,
    title: str,
    n_min: int,
    n_max: int,
    height: int = 320,
) -> alt.Chart:

    x = d_both[(d_both["n_q"] >= int(n_min)) & (d_both["n_q"] <= int(n_max))].copy()
    if x.empty:
        return alt.Chart(pd.DataFrame({"_": ["Keine Daten in diesem Bereich."]})).mark_text().encode(text="_")

    label_angle = 0 if (n_max - n_min) <= 12 else 90

    # --- Y-Achse: feste Domain + custom ticks (robust über n_min/n_max) ---
    is_brutto = "brutto" in (title or "").lower()

    # default domains
    if is_brutto:
        y_domain = [-0.05, 0.30]
        y_ticks = [-0.05, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
        # für 4–12 mehr nach oben
        if n_min <= 4 and n_max <= 12:
            y_ticks += [0.15, 0.20, 0.25, 0.30]
        # für 31–60 fein im unteren Bereich (optional)
        if n_min >= 31:
            y_ticks += [0.02, 0.04, 0.06, 0.08]
    else:
        # Netto
        y_domain = [-0.05, 0.25 ]
        y_ticks = [-0.10, -0.05, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
        # für 4–12 mehr nach oben
        if n_min <= 4 and n_max <= 12:
            y_ticks += [0.15, 0.20, 0.25]
        # für 31–60 fein (optional)
        if n_min >= 31:
            y_ticks += [0.02, 0.04, 0.06]

    # wichtig: sort + dedupe + innerhalb Domain
    y_ticks = sorted({t for t in y_ticks if y_domain[0] <= t <= y_domain[1]})

    # --- Chart ---
    return (
        alt.Chart(x)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("n_q:O", title="Haltedauer (Quartale)", axis=alt.Axis(labelAngle=label_angle)),
            y=alt.Y(
                "value:Q",
                title="Ø Quartalsrendite (geom.)",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(format=".1%", values=y_ticks),
            ),
            xOffset=alt.XOffset("mode:N"),
            color=alt.Color("mode:N", title="Modus"),
            tooltip=[
                alt.Tooltip("mode:N", title="Modus"),
                alt.Tooltip("n_q:O", title="n_q"),
                alt.Tooltip("value:Q", title="Rendite", format=".2%"),
                alt.Tooltip("buy_q:O", title="Buy"),
                alt.Tooltip("sell_q:O", title="Sell"),
            ],
        )
        .properties(title=title, height=height)
    )





# ============================================================
# Saved NAV
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

    df_nav, _ = build_saved_buyhold_series_with_liquidation(
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

    return df_nav, {"status": "ok", "name": doc.get("name", "")}


# ============================================================
# Detail view: welche Werte stecken hinter n_q?
# ============================================================
def show_values_for_nq(tri: pd.DataFrame, n_q_target: int, *, sort_desc: bool = True) -> pd.DataFrame:
    if tri is None or tri.empty:
        return pd.DataFrame(columns=["buy_q", "sell_q", "value", "n_q"])

    d = tri.copy()
    d["n_q"] = pd.to_numeric(d.get("n_q"), errors="coerce")
    d["value"] = pd.to_numeric(d.get("value"), errors="coerce")
    d = d.dropna(subset=["n_q", "value"]).copy()
    d["n_q"] = d["n_q"].astype(int)

    out = d[d["n_q"] == int(n_q_target)].copy()
    if out.empty:
        return pd.DataFrame(columns=["buy_q", "sell_q", "value", "n_q"])

    out = out.sort_values("value", ascending=not sort_desc).reset_index(drop=True)
    cols = [c for c in ["buy_q", "sell_q", "value", "n_q"] if c in out.columns]
    return out[cols]


# ============================================================
# Saved Triangle Builder (netto oder brutto) [intern, nicht anzeigen]
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def build_saved_triangle(
    q_start: str,
    q_end: str,
    end_date_cutoff: pd.Timestamp,
    base_capital: float,
    tax_rate: float,
    apply_tax: bool,
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

    for p in ports_q:
        pid = p["id"]
        df_nav, meta = compute_saved_portfolio_nav(pid, base_capital=base_capital, end_date=end_dt)
        status = meta.get("status", "unknown")
        status_rows.append({"name": p["name"], "id": pid, "status": status})
        if status == "ok" and df_nav is not None and not df_nav.empty:
            nav_cache[pid] = df_nav
            buy_q_by_pid[pid] = p["name"]

    tax_r = max(0.0, min(1.0, float(tax_rate)))

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

            if apply_tax:
                gain = max(0.0, v1_brutto - v0)
                v1 = v1_brutto - (gain * tax_r)
            else:
                v1 = v1_brutto

            val = geom_avg_quarter_return(v0, v1, n_q)
            if pd.isna(val):
                continue

            rows.append({"buy_q": buy_q, "sell_q": sell_q, "n_q": n_q, "value": float(val)})

    return pd.DataFrame(rows), pd.DataFrame(status_rows)


# ============================================================
# Benchmark triangles (intern, NICHT anzeigen) -> Diagonalen
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def build_benchmark_triangle_endonly_tax(
    df_prices: pd.DataFrame,
    tax_rate: float,
    apply_tax: bool,
    *,
    b_from: date,
    b_to: date,
) -> pd.DataFrame:
    """
    Baut intern Zellen (buy_q, sell_q, n_q, value) aus EOQ-Endwerten.
    Netto-Logik: Steuer nur beim Verkauf auf Gewinn seit Kauf.
    """
    if df_prices is None or df_prices.empty:
        return pd.DataFrame()

    x = df_prices.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["adjClose"] = pd.to_numeric(x["adjClose"], errors="coerce")
    x = x.dropna(subset=["date", "adjClose"]).sort_values("date")

    mask = (x["date"].dt.date >= b_from) & (x["date"].dt.date <= b_to)
    x = x.loc[mask].copy()
    if x.empty or len(x) < 2:
        return pd.DataFrame()

    eoq = build_quarter_eoq_series_from_daily(x[["date", "adjClose"]].copy(), "adjClose")
    if eoq.empty or len(eoq) < 2:
        return pd.DataFrame()

    df_like = eoq[["quarter", "end_value"]].copy()
    df_like["quarter"] = df_like["quarter"].astype(str)
    df_like["end_value"] = pd.to_numeric(df_like["end_value"], errors="coerce")
    df_like = df_like.dropna(subset=["quarter", "end_value"])
    df_like = df_like[df_like["end_value"] > 0].copy()

    df_like["q_index"] = df_like["quarter"].map(quarter_to_index)
    df_like = df_like.dropna(subset=["q_index"]).copy()
    df_like["q_index"] = df_like["q_index"].astype(int)
    df_like = df_like.sort_values("q_index").reset_index(drop=True)

    tax_r = max(0.0, min(1.0, float(tax_rate)))

    q = df_like["quarter"].tolist()
    qi = df_like["q_index"].to_numpy()
    nav = df_like["end_value"].astype(float).to_numpy()

    rows: List[Dict[str, Any]] = []
    n = len(df_like)
    for i in range(n):
        for j in range(i + 1, n):
            n_q = int(qi[j] - qi[i])
            if n_q <= 0:
                continue

            v0 = float(nav[i])
            v1_brutto = float(nav[j])

            if apply_tax:
                gain = max(0.0, v1_brutto - v0)
                v1 = v1_brutto - (gain * tax_r)
            else:
                v1 = v1_brutto

            value = geom_avg_quarter_return(v0, v1, n_q)
            if pd.isna(value):
                continue

            rows.append({"buy_q": q[i], "sell_q": q[j], "value": float(value), "n_q": n_q})

    return pd.DataFrame(rows)


# ============================================================
# Diagonalen (Gruppierung nach Haltedauer n_q)
# ============================================================
def extract_diagonals(tri: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if tri is None or tri.empty:
        return pd.DataFrame(columns=["n_q", "buy_q", "sell_q", "value"])

    d = tri.copy()

    if "n_q" not in d.columns:
        d["buy_i"] = d["buy_q"].map(quarter_to_index)
        d["sell_i"] = d["sell_q"].map(quarter_to_index)
        d = d.dropna(subset=["buy_i", "sell_i"]).copy()
        d["n_q"] = (d["sell_i"] - d["buy_i"]).astype(int)

    d["n_q"] = pd.to_numeric(d["n_q"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["n_q", "value"]).copy()
    d["n_q"] = d["n_q"].astype(int)

    d = d[(d["n_q"] >= int(MIN_NQ)) & (d["n_q"] <= int(max_n))].copy()
    return d[["n_q", "buy_q", "sell_q", "value"]].sort_values(["n_q", "buy_q", "sell_q"]).reset_index(drop=True)


# ============================================================
# Dynamik -> Triangle (intern)
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def compute_dynamic_triangle(_es_client, initial_capital: float, tax_rate: float, y0: int, y1: int) -> pd.DataFrame:
    if not HAS_DYNAMIC:
        return pd.DataFrame()

    portfolios = list_portfolios(_es_client) or []
    minimal = [{"id": p.get("id"), "name": p.get("name", "")} for p in portfolios if p.get("id")]

    df_dyn = simulate_dynamic_cached(
        _es_client=_es_client,
        portfolio_minimal=minimal,
        initial_capital=float(initial_capital),
        tax_rate=float(tax_rate),
        year_from=int(y0),
        year_to=int(y1),
        prices_index=PRICE_INDEX,
        stocks_index=STOCKS_INDEX,
    )
    if df_dyn is None or df_dyn.empty:
        return pd.DataFrame()

    df_like = df_dyn[["quarter", "end_value"]].copy()
    df_like["quarter"] = df_like["quarter"].astype(str)
    df_like["end_value"] = pd.to_numeric(df_like["end_value"], errors="coerce")
    df_like = df_like.dropna(subset=["quarter", "end_value"])
    df_like = df_like[df_like["end_value"] > 0].copy()

    df_like["q_index"] = df_like["quarter"].map(quarter_to_index)
    df_like = df_like.dropna(subset=["q_index"]).copy()
    df_like["q_index"] = df_like["q_index"].astype(int)
    df_like = df_like.sort_values("q_index").reset_index(drop=True)

    q = df_like["quarter"].tolist()
    qi = df_like["q_index"].to_numpy()
    nav = df_like["end_value"].astype(float).to_numpy()

    rows = []
    n = len(df_like)
    for i in range(n):
        for j in range(i + 1, n):
            n_q = int(qi[j] - qi[i])
            if n_q <= 0:
                continue
            v0, v1 = float(nav[i]), float(nav[j])
            value = geom_avg_quarter_return(v0, v1, n_q)
            if pd.isna(value):
                continue
            rows.append({"buy_q": q[i], "sell_q": q[j], "value": float(value), "n_q": n_q})

    return pd.DataFrame(rows)


# ============================================================
# UI Settings (Sidebar)
# ============================================================
with st.sidebar:
    st.header("Basis – Settings")
    q_start = st.text_input("q_start (YYYY-Qx)", value="2010-Q1")
    q_end = st.text_input("q_end (YYYY-Qx)", value="2025-Q4")
    max_n = st.number_input("max_n (Boxplots)", min_value=1, max_value=120, value=60, step=1)
    end_cutoff = pd.to_datetime(st.date_input("end_cutoff", value=END_DATE_DEFAULT.date()))
    base_capital = st.number_input("base_capital (nur Skalierung)", min_value=1.0, value=1000.0, step=100.0)

    st.divider()
    st.header("Steuern")
    tax_rate_saved = st.number_input("tax_rate Saved (für Netto)", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    dyn_tax = st.number_input("tax_rate Dynamik (für Netto)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="dyn_tax")

    st.divider()
    st.header("Benchmarks – Settings")
    bench_from = st.date_input("Benchmark von", value=date(2010, 1, 1), key="bench_from")
    bench_to = st.date_input("Benchmark bis", value=date.today(), key="bench_to")
    tax_rate_bench = st.number_input("tax_rate Benchmarks (für Netto)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, key="bench_tax")

    st.divider()
    st.header("Dynamik – Settings")
    dyn_initial = st.number_input("Startkapital (USD)", min_value=0.0, value=1000.0, step=100.0, key="dyn_init")
    dyn_year_from = st.number_input("Von Jahr (Dynamik)", min_value=1990, max_value=2100, value=2010, step=1, key="dyn_y0")
    dyn_year_to = st.number_input("Bis Jahr (Dynamik)", min_value=1990, max_value=2100, value=2025, step=1, key="dyn_y1")

    st.divider()
    st.header("Anzeige (ein/aus)")
    show_saved_net = st.checkbox("Saved Netto (mit Steuer)", True)
    show_saved_gross = st.checkbox("Saved Brutto (ohne Steuer)", True)
    show_dyn_net = st.checkbox("Dynamik Netto (mit Steuer)", True)
    show_dyn_gross = st.checkbox("Dynamik Brutto (ohne Steuer)", True)

    st.divider()
    st.header("Benchmarks anzeigen")
    show_bench_net = st.checkbox("GSPC/SPXEW Netto (mit Steuer)", True)
    show_bench_gross = st.checkbox("GSPC/SPXEW Brutto (ohne Steuer)", True)
    show_bench_lines = st.checkbox("Benchmarks im Linienchart berücksichtigen", True)

    st.divider()
    show_counts = st.checkbox("Counts-Bars anzeigen", True)
    show_details = st.checkbox("Detail-Tabellen anzeigen", True)
    show_median_lines = st.checkbox("Median-Linien anzeigen", True)

    st.divider()
    st.header("Linienchart (Settings)")
    line_max_n = st.number_input(
        "Linienchart bis Haltedauer (n_q)",
        min_value=MIN_NQ,
        max_value=120,
        value=max(60, MIN_NQ),
        step=1,
        key="line_max_n",
    )

    line_split_mode = st.selectbox(
        "Linienchart Splits",
        [f"gesplittet ({MIN_NQ}–12, 13–30, 31–max)", "ein Chart"],
        index=0,
    )

    st.divider()
    st.header("Vergleich (Saved vs Dynamik)")
    show_compare_net = st.checkbox("Vergleich Netto (Saved vs Dynamik)", True)
    show_compare_gross = st.checkbox("Vergleich Brutto (Saved vs Dynamik)", True)
    compare_max = st.number_input(
        "Vergleich bis Haltedauer (n_q)",
        min_value=MIN_NQ,
        max_value=60,
        value=max(60, MIN_NQ),
        step=1,
        key="cmp_max",
    )

# ============================================================
# Global ranges / splits (AFTER sidebar!)  <-- FIX für cmp_max
# ============================================================
split_a_min, split_a_max = MIN_NQ, 12
split_b_min, split_b_max = 13, min(int(max_n), 60)

cmp_max = int(min(int(compare_max), int(max_n), 60))
cmp_splits = [(MIN_NQ, 12), (13, 30), (31, 60)]


# ============================================================
# Rendering helper (untereinander)
# ============================================================
def render_block(
    title_prefix: str,
    tri: pd.DataFrame,
    d: pd.DataFrame,
    max_n_local: int,
    key_prefix: str,
) -> None:
    st.subheader(f"Boxplots ({title_prefix}) – gesplittet (untereinander)")

    # MIN_NQ–12
    st.altair_chart(
        plot_boxplots_by_nq_range(d, f"{title_prefix} – Haltedauer {split_a_min}–{split_a_max}", split_a_min, split_a_max),
        use_container_width=True,
    )
    if show_counts:
        st.altair_chart(
            plot_counts_by_nq_range(d, f"{title_prefix} – Anzahl Werte {split_a_min}–{split_a_max}", split_a_min, split_a_max),
            use_container_width=True,
        )
    if show_median_lines:
        med = median_by_nq(d, title_prefix)
        st.altair_chart(
            plot_median_lines(med, f"{title_prefix} – Median-Linie {split_a_min}–{split_a_max}", split_a_min, split_a_max, height=240),
            use_container_width=True,
        )

    st.write("")

    # 13–60
    st.altair_chart(
        plot_boxplots_by_nq_range(d, f"{title_prefix} – Haltedauer {split_b_min}–{split_b_max}", split_b_min, split_b_max),
        use_container_width=True,
    )
    if show_counts:
        st.altair_chart(
            plot_counts_by_nq_range(d, f"{title_prefix} – Anzahl Werte {split_b_min}–{split_b_max}", split_b_min, split_b_max),
            use_container_width=True,
        )
    if show_median_lines:
        med = median_by_nq(d, title_prefix)
        st.altair_chart(
            plot_median_lines(med, f"{title_prefix} – Median-Linie {split_b_min}–{split_b_max}", split_b_min, split_b_max, height=240),
            use_container_width=True,
        )

    if show_details:
        st.subheader(f"Detailansicht ({title_prefix}) – Werte hinter n_q (absteigend)")
        nq_dbg = st.number_input(
            f"Welche Haltedauer anzeigen? ({title_prefix})",
            min_value=int(MIN_NQ),
            max_value=int(max_n_local),
            value=int(MIN_NQ),
            key=f"{key_prefix}_nq",
        )
        detail = show_values_for_nq(tri, int(nq_dbg), sort_desc=True)
        st.write(f"Anzahl Werte: {len(detail)}")
        st.dataframe(detail, use_container_width=True, hide_index=True)


def render_compare_section(title: str, d_saved: pd.DataFrame, d_dyn: pd.DataFrame) -> None:
    st.subheader(title)
    if d_saved.empty or d_dyn.empty:
        st.warning("Vergleich nicht möglich (Saved oder Dynamik ist leer).")
        return

    both = stack_saved_dyn(d_saved, d_dyn)
    both = both[both["n_q"] >= int(MIN_NQ)].copy()

    for a, b in cmp_splits:
        if cmp_max < a:
            continue
        hi = min(b, cmp_max)
        st.altair_chart(
            plot_boxplots_saved_vs_dyn(both, f"{title} – Haltedauer {a}–{hi}", a, hi, height=340),
            use_container_width=True,
        )

    if show_median_lines:
        med_saved = median_by_nq(d_saved, "Saved")
        med_dyn = median_by_nq(d_dyn, "Dynamik")
        med_both = pd.concat([med_saved, med_dyn], ignore_index=True)

        for a, b in cmp_splits:
            if cmp_max < a:
                continue
            hi = min(b, cmp_max)
            st.altair_chart(
                plot_median_lines(med_both, f"{title} – Median-Linien {a}–{hi}", a, hi, height=260),
                use_container_width=True,
            )


# ============================================================
# Build all triangles (intern) + Diagonalen
# ============================================================
tri_saved_net = pd.DataFrame()
tri_saved_gross = pd.DataFrame()
tri_dyn_net = pd.DataFrame()
tri_dyn_gross = pd.DataFrame()

tri_gspc_net = pd.DataFrame()
tri_gspc_gross = pd.DataFrame()
tri_spxew_net = pd.DataFrame()
tri_spxew_gross = pd.DataFrame()

d_saved_net = pd.DataFrame()
d_saved_gross = pd.DataFrame()
d_dyn_net = pd.DataFrame()
d_dyn_gross = pd.DataFrame()

d_gspc_net = pd.DataFrame()
d_gspc_gross = pd.DataFrame()
d_spxew_net = pd.DataFrame()
d_spxew_gross = pd.DataFrame()

# -------------------------
# SAVED
# -------------------------
need_saved = show_saved_net or show_saved_gross or show_compare_net or show_compare_gross
if need_saved:
    if show_saved_net or show_compare_net:
        tri_saved_net, _ = build_saved_triangle(
            q_start=q_start,
            q_end=q_end,
            end_date_cutoff=end_cutoff,
            base_capital=float(base_capital),
            tax_rate=float(tax_rate_saved),
            apply_tax=True,
        )
        d_saved_net = extract_diagonals(tri_saved_net, max_n=int(max_n)) if not tri_saved_net.empty else pd.DataFrame()

    if show_saved_gross or show_compare_gross:
        tri_saved_gross, _ = build_saved_triangle(
            q_start=q_start,
            q_end=q_end,
            end_date_cutoff=end_cutoff,
            base_capital=float(base_capital),
            tax_rate=0.0,
            apply_tax=False,
        )
        d_saved_gross = extract_diagonals(tri_saved_gross, max_n=int(max_n)) if not tri_saved_gross.empty else pd.DataFrame()

# -------------------------
# DYNAMIK
# -------------------------
need_dyn = show_dyn_net or show_dyn_gross or show_compare_net or show_compare_gross
if need_dyn:
    if not HAS_DYNAMIC:
        st.warning("simulate_dynamic_cached ist nicht verfügbar (Import fehlgeschlagen). Dynamik-Teil wird übersprungen.")
    else:
        if show_dyn_net or show_compare_net:
            tri_dyn_net = compute_dynamic_triangle(es, dyn_initial, float(dyn_tax), int(dyn_year_from), int(dyn_year_to))
            d_dyn_net = extract_diagonals(tri_dyn_net, max_n=int(max_n)) if not tri_dyn_net.empty else pd.DataFrame()

        if show_dyn_gross or show_compare_gross:
            tri_dyn_gross = compute_dynamic_triangle(es, dyn_initial, 0.0, int(dyn_year_from), int(dyn_year_to))
            d_dyn_gross = extract_diagonals(tri_dyn_gross, max_n=int(max_n)) if not tri_dyn_gross.empty else pd.DataFrame()

# -------------------------
# BENCHMARKS (GSPC / SPXEW)
# -------------------------
need_bench = show_bench_net or show_bench_gross or show_bench_lines
if need_bench:
    df_gspc = load_gspc()
    df_spxew = load_spxew()

    if show_bench_net or show_bench_lines:
        tri_gspc_net = build_benchmark_triangle_endonly_tax(
            df_gspc, tax_rate=float(tax_rate_bench), apply_tax=True, b_from=bench_from, b_to=bench_to
        )
        d_gspc_net = extract_diagonals(tri_gspc_net, max_n=int(max_n)) if not tri_gspc_net.empty else pd.DataFrame()

        tri_spxew_net = build_benchmark_triangle_endonly_tax(
            df_spxew, tax_rate=float(tax_rate_bench), apply_tax=True, b_from=bench_from, b_to=bench_to
        )
        d_spxew_net = extract_diagonals(tri_spxew_net, max_n=int(max_n)) if not tri_spxew_net.empty else pd.DataFrame()

    if show_bench_gross or show_bench_lines:
        tri_gspc_gross = build_benchmark_triangle_endonly_tax(
            df_gspc, tax_rate=0.0, apply_tax=False, b_from=bench_from, b_to=bench_to
        )
        d_gspc_gross = extract_diagonals(tri_gspc_gross, max_n=int(max_n)) if not tri_gspc_gross.empty else pd.DataFrame()

        tri_spxew_gross = build_benchmark_triangle_endonly_tax(
            df_spxew, tax_rate=0.0, apply_tax=False, b_from=bench_from, b_to=bench_to
        )
        d_spxew_gross = extract_diagonals(tri_spxew_gross, max_n=int(max_n)) if not tri_spxew_gross.empty else pd.DataFrame()

# ============================================================
# Linienchart: Median-Rendite vs Haltedauer
# ============================================================
st.divider()
st.header("Linienchart: Median-Rendite über Haltedauer")

line_max = int(min(int(line_max_n), int(max_n), 120))
med_frames: List[pd.DataFrame] = []

# Saved/Hold
if not d_saved_net.empty:
    med_frames.append(median_by_nq(d_saved_net, "Hold Netto (mit Steuer)"))
if not d_saved_gross.empty:
    med_frames.append(median_by_nq(d_saved_gross, "Hold Brutto (ohne Steuer)"))

# Dynamik
if not d_dyn_net.empty:
    med_frames.append(median_by_nq(d_dyn_net, "Dynamik Netto (mit Steuer)"))
if not d_dyn_gross.empty:
    med_frames.append(median_by_nq(d_dyn_gross, "Dynamik Brutto (ohne Steuer)"))

# Benchmarks (optional)
if show_bench_lines:
    if not d_gspc_net.empty:
        med_frames.append(median_by_nq(d_gspc_net, "GSPC Netto (mit Steuer)"))
    if not d_gspc_gross.empty:
        med_frames.append(median_by_nq(d_gspc_gross, "GSPC Brutto (ohne Steuer)"))
    if not d_spxew_net.empty:
        med_frames.append(median_by_nq(d_spxew_net, "SPXEW Netto (mit Steuer)"))
    if not d_spxew_gross.empty:
        med_frames.append(median_by_nq(d_spxew_gross, "SPXEW Brutto (ohne Steuer)"))

med_all = pd.concat(med_frames, ignore_index=True) if med_frames else pd.DataFrame()

if med_all.empty:
    st.warning("Keine Medians verfügbar (alle Diagonalen leer).")
else:
    available_series = sorted(med_all["mode"].dropna().unique().tolist())

    with st.expander("Linienchart – aktive Serien", expanded=False):
        default_selected = st.session_state.get("selected_line_series_main", available_series)
        default_selected = [s for s in default_selected if s in available_series]
        if not default_selected:
            default_selected = available_series

        selected_line_series_main = st.multiselect(
            "Welche Serien sollen im Linienchart angezeigt werden?",
            options=available_series,
            default=default_selected,
            key="selected_line_series_main",
        )

    selected = [s for s in selected_line_series_main if s in available_series]
    if not selected:
        selected = available_series

    med_all = med_all[med_all["mode"].isin(selected)].copy()

    if line_split_mode.startswith("gesplittet"):
        line_splits = [(MIN_NQ, 12), (13, 30), (31, line_max)]
        for a, b in line_splits:
            if line_max < a:
                continue
            hi = min(b, line_max)
            st.altair_chart(
                plot_median_lines_multi(
                    med_all,
                    f"Median-Linien – Haltedauer {a}–{hi}",
                    a,
                    hi,
                    height=340,
                ),
                use_container_width=True,
            )
    else:
        st.altair_chart(
            plot_median_lines_multi(
                med_all,
                f"Median-Linien – Haltedauer {MIN_NQ}–{line_max}",
                MIN_NQ,
                line_max,
                height=380,
            ),
            use_container_width=True,
        )

# ============================================================
# OUTPUT: Saved Netto / Brutto
# ============================================================
if show_saved_net:
    st.divider()
    st.header("Saved Portfolios – Netto (MIT Steuer)")
    if tri_saved_net.empty:
        st.warning("Saved Netto intern leer. Check Quartalsrange / Daten.")
    else:
        st.caption(f"Diagonal rows: {len(d_saved_net)}")
        render_block("Saved Netto", tri_saved_net, d_saved_net, int(max_n), "saved_net")

if show_saved_gross:
    st.divider()
    st.header("Saved Portfolios – Brutto (OHNE Steuer)")
    if tri_saved_gross.empty:
        st.warning("Saved Brutto intern leer. Check Quartalsrange / Daten.")
    else:
        st.caption(f"Diagonal rows: {len(d_saved_gross)}")
        render_block("Saved Brutto", tri_saved_gross, d_saved_gross, int(max_n), "saved_gross")

# ============================================================
# OUTPUT: Dynamik Netto / Brutto
# ============================================================
if show_dyn_net:
    st.divider()
    st.header("Dynamik (Quartals-Umstieg) – Netto (MIT Steuer)")
    if tri_dyn_net.empty:
        st.warning("Dynamik Netto intern leer (zu wenig Quartale / keine Daten).")
    else:
        st.caption(f"Diagonal rows: {len(d_dyn_net)}")
        render_block("Dynamik Netto", tri_dyn_net, d_dyn_net, int(max_n), "dyn_net")

if show_dyn_gross:
    st.divider()
    st.header("Dynamik (Quartals-Umstieg) – Brutto (OHNE Steuer)")
    if tri_dyn_gross.empty:
        st.warning("Dynamik Brutto intern leer (zu wenig Quartale / keine Daten).")
    else:
        st.caption(f"Diagonal rows: {len(d_dyn_gross)}")
        render_block("Dynamik Brutto", tri_dyn_gross, d_dyn_gross, int(max_n), "dyn_gross")

# ============================================================
# OUTPUT: Benchmarks (untereinander, NICHT im Saved-vs-Dyn-Vergleich)
# ============================================================
if show_bench_net or show_bench_gross:
    st.divider()
    st.header("Benchmarks – Boxplots & Median-Linien (untereinander)")
    st.caption(f"Zeitraum: {bench_from} → {bench_to} | Steuer (Netto): {float(tax_rate_bench)*100:.0f}% (nur Endverkauf)")

    if show_bench_net:
        st.subheader("GSPC – Netto (MIT Steuer)")
        if tri_gspc_net.empty:
            st.warning("GSPC Netto intern leer (Check BENCH_INDEX / Zeitraum).")
        else:
            st.caption(f"Diagonal rows: {len(d_gspc_net)}")
            render_block("GSPC Netto", tri_gspc_net, d_gspc_net, int(max_n), "gspc_net")

        st.subheader("SPXEW – Netto (MIT Steuer)")
        if tri_spxew_net.empty:
            st.warning("SPXEW Netto intern leer (Check BENCH_INDEX / Zeitraum).")
        else:
            st.caption(f"Diagonal rows: {len(d_spxew_net)}")
            render_block("SPXEW Netto", tri_spxew_net, d_spxew_net, int(max_n), "spxew_net")

    if show_bench_gross:
        st.subheader("GSPC – Brutto (OHNE Steuer)")
        if tri_gspc_gross.empty:
            st.warning("GSPC Brutto intern leer (Check BENCH_INDEX / Zeitraum).")
        else:
            st.caption(f"Diagonal rows: {len(d_gspc_gross)}")
            render_block("GSPC Brutto", tri_gspc_gross, d_gspc_gross, int(max_n), "gspc_gross")

        st.subheader("SPXEW – Brutto (OHNE Steuer)")
        if tri_spxew_gross.empty:
            st.warning("SPXEW Brutto intern leer (Check BENCH_INDEX / Zeitraum).")
        else:
            st.caption(f"Diagonal rows: {len(d_spxew_gross)}")
            render_block("SPXEW Brutto", tri_spxew_gross, d_spxew_gross, int(max_n), "spxew_gross")

# ============================================================
# OUTPUT: Vergleich Saved vs Dynamik (nebeneinander) – Benchmarks NICHT enthalten
# ============================================================
st.divider()
st.header("Vergleich: Saved vs Dynamik (nebeneinander) – bis n_q (untereinander)")
st.caption("Benchmarks werden hier bewusst NICHT mit aufgenommen.")

if cmp_max < MIN_NQ:
    st.info(f"Vergleich benötigt compare_max >= {MIN_NQ}.")
else:
    if show_compare_net:
        render_compare_section("Vergleich – Netto (mit Steuer)", d_saved_net, d_dyn_net)
    if show_compare_gross:
        render_compare_section("Vergleich – Brutto (ohne Steuer)", d_saved_gross, d_dyn_gross)
