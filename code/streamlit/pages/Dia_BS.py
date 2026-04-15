from __future__ import annotations

import io
import os
import sys
import zipfile
from datetime import date
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------
# Einheitliche Schriftgrößen
# ------------------------------------------------------------
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 22
LEGEND_FONTSIZE = 12

plt.rcParams.update(
    {
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
    }
)

# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(page_title="Liniencharts Matplotlib", layout="wide")
st.title("Liniencharts mit Matplotlib")

# ------------------------------------------------------------
# Pfad-Setup
# ------------------------------------------------------------
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.funktionen import get_es_connection, list_portfolios, load_portfolio  # noqa: E402
from src.portfolio_simulation import (  # noqa: E402
    build_saved_buyhold_series_with_liquidation,
    geom_avg_quarter_return,
    get_buy_date_like_dynamic_for_portfolio,
    list_quarters_between,
    parse_portfolio_name,
    portfolio_doc_to_amount_weights,
    quarter_end_ts,
    quarter_to_index,
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
        frames.append(
            d[["date", "adjClose"]]
            .rename(columns={"adjClose": sym})
            .set_index("date")
        )

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
# EOQ helper
# ============================================================
def build_quarter_eoq_series_from_daily(df_daily: pd.DataFrame, value_col: str) -> pd.DataFrame:
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
# Hold / Saved NAV
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
# Triangles
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def build_saved_triangle(
    q_start: str,
    q_end: str,
    end_date_cutoff: pd.Timestamp,
    base_capital: float,
    tax_rate: float,
    apply_tax: bool,
) -> pd.DataFrame:
    ports = list_portfolios(es) or []
    ports_q: List[Dict[str, Any]] = []

    for p in ports:
        name = p.get("name", "")
        pid = p.get("id")
        if pid and parse_portfolio_name(name) is not None:
            ports_q.append({"id": pid, "name": name})

    sell_quarters = list_quarters_between(q_start, q_end)
    if not sell_quarters:
        return pd.DataFrame()

    end_dt_qend = quarter_end_ts(q_end)
    if end_dt_qend is None:
        return pd.DataFrame()

    end_dt = min(pd.Timestamp(end_dt_qend), pd.Timestamp(end_date_cutoff))

    nav_cache: Dict[str, pd.DataFrame] = {}
    buy_q_by_pid: Dict[str, str] = {}

    for p in ports_q:
        pid = p["id"]
        df_nav, meta = compute_saved_portfolio_nav(
            pid,
            base_capital=base_capital,
            end_date=end_dt,
        )
        status = meta.get("status", "unknown")
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

            rows.append(
                {
                    "buy_q": buy_q,
                    "sell_q": sell_q,
                    "n_q": n_q,
                    "value": float(val),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=True, ttl=600)
def build_benchmark_triangle_endonly_tax(
    df_prices: pd.DataFrame,
    tax_rate: float,
    apply_tax: bool,
    *,
    b_from: date,
    b_to: date,
) -> pd.DataFrame:
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

            rows.append(
                {
                    "buy_q": q[i],
                    "sell_q": q[j],
                    "value": float(value),
                    "n_q": n_q,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=True, ttl=600)
def compute_dynamic_triangle(
    _es_client,
    initial_capital: float,
    tax_rate: float,
    y0: int,
    y1: int,
) -> pd.DataFrame:
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

    rows: List[Dict[str, Any]] = []
    n = len(df_like)

    for i in range(n):
        for j in range(i + 1, n):
            n_q = int(qi[j] - qi[i])
            if n_q <= 0:
                continue

            v0 = float(nav[i])
            v1 = float(nav[j])
            value = geom_avg_quarter_return(v0, v1, n_q)
            if pd.isna(value):
                continue

            rows.append(
                {
                    "buy_q": q[i],
                    "sell_q": q[j],
                    "value": float(value),
                    "n_q": n_q,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Helpers
# ============================================================
def extract_diagonals(tri: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if tri is None or tri.empty:
        return pd.DataFrame(columns=["n_q", "buy_q", "sell_q", "value"])

    d = tri.copy()
    d["n_q"] = pd.to_numeric(d["n_q"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["n_q", "value"]).copy()
    d["n_q"] = d["n_q"].astype(int)

    d = d[(d["n_q"] >= int(MIN_NQ)) & (d["n_q"] <= int(max_n))].copy()
    return d[["n_q", "buy_q", "sell_q", "value"]].sort_values(
        ["n_q", "buy_q", "sell_q"]
    ).reset_index(drop=True)


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
    med["mode"] = label
    return med


def build_tax_effect_df(brutto_df: pd.DataFrame, netto_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if brutto_df is None or netto_df is None or brutto_df.empty or netto_df.empty:
        return pd.DataFrame(columns=["n_q", "median", "mode"])

    b = median_by_nq(brutto_df, label)[["n_q", "median"]].rename(columns={"median": "brutto"})
    n = median_by_nq(netto_df, label)[["n_q", "median"]].rename(columns={"median": "netto"})

    x = b.merge(n, on="n_q", how="inner")
    if x.empty:
        return pd.DataFrame(columns=["n_q", "median", "mode"])

    x["median"] = x["brutto"] - x["netto"]
    x["mode"] = label
    return x[["n_q", "median", "mode"]].sort_values("n_q").reset_index(drop=True)


def build_violin_df(
    d_dyn_gross: pd.DataFrame,
    d_dyn_net: pd.DataFrame,
    d_hold_gross: pd.DataFrame,
    d_hold_net: pd.DataFrame,
    n_min: int,
    n_max: int,
) -> pd.DataFrame:
    rows = []

    def _append(df: pd.DataFrame, label: str) -> None:
        if df is None or df.empty:
            return

        x = df.copy()
        x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
        x["value"] = pd.to_numeric(x["value"], errors="coerce")
        x = x.dropna(subset=["n_q", "value"]).copy()
        x["n_q"] = x["n_q"].astype(int)
        x = x[(x["n_q"] >= int(n_min)) & (x["n_q"] <= int(n_max))].copy()

        for v in x["value"].tolist():
            rows.append(
                {
                    "Strategie / Variante": label,
                    "Rendite": float(v),
                }
            )

    _append(d_dyn_gross, "Dynamik Brutto")
    _append(d_dyn_net, "Dynamik Netto")
    _append(d_hold_gross, "Hold Brutto")
    _append(d_hold_net, "Hold Netto")

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    order = ["Dynamik Brutto", "Dynamik Netto", "Hold Brutto", "Hold Netto"]
    out["Strategie / Variante"] = pd.Categorical(
        out["Strategie / Variante"],
        categories=order,
        ordered=True,
    )
    return out.sort_values("Strategie / Variante").reset_index(drop=True)


def plot_violin_matplotlib(
    violin_df: pd.DataFrame,
    title: str,
    figsize: tuple = (14, 8),
):
    if violin_df is None or violin_df.empty:
        return None

    order = ["Dynamik Brutto", "Dynamik Netto", "Hold Brutto", "Hold Netto"]

    data = []
    labels = []
    for label in order:
        vals = violin_df.loc[
            violin_df["Strategie / Variante"] == label,
            "Rendite",
        ].dropna().tolist()
        if vals:
            data.append(vals)
            labels.append(label)

    if not data:
        return None

    color_map = {
        "Dynamik Brutto": "#5E88D8",
        "Dynamik Netto": "#A9C9F5",
        "Hold Brutto": "#E57373",
        "Hold Netto": "#E6B8B8",
    }

    fig, ax = plt.subplots(figsize=figsize)

    vp = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    for body, label in zip(vp["bodies"], labels):
        body.set_facecolor(color_map.get(label, "#999999"))
        body.set_edgecolor("#666666")
        body.set_alpha(1.0)
        body.set_linewidth(1.5)

    ax.boxplot(
        data,
        positions=range(1, len(data) + 1),
        widths=0.08,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2),
        boxprops=dict(facecolor="#555555", color="#555555", linewidth=1.5),
        whiskerprops=dict(color="#555555", linewidth=1.2),
        capprops=dict(color="#555555", linewidth=1.2),
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Strategie / Variante", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Ø Quartalsrendite (geom.)", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    return fig


def plot_median_lines_matplotlib(
    med_all: pd.DataFrame,
    title: str,
    n_min: int,
    n_max: int,
    figsize: tuple = (16, 6.5),
):
    if med_all is None or med_all.empty:
        return None

    x = med_all.copy()
    x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
    x["median"] = pd.to_numeric(x["median"], errors="coerce")
    x = x.dropna(subset=["n_q", "median"]).copy()
    x["n_q"] = x["n_q"].astype(int)
    x = x[(x["n_q"] >= n_min) & (x["n_q"] <= n_max)].copy()

    if x.empty:
        return None

    color_map = {
        "Dynamik Netto (mit Steuer)": "#1565C0",
        "GSPC Netto (mit Steuer)": "#90CAF9",
        "Hold Netto (mit Steuer)": "#EF5350",
        "SPXEW Netto (mit Steuer)": "#F8BBD0",
        "Dynamik Brutto (ohne Steuer)": "#1565C0",
        "GSPC Brutto (ohne Steuer)": "#90CAF9",
        "Hold Brutto (ohne Steuer)": "#EF5350",
        "SPXEW Brutto (ohne Steuer)": "#F8BBD0",
        "Dynamik": "#1565C0",
        "Hold": "#66BB6A",
        "GSPC": "#EF5350",
        "SPXEW": "#F9A825",
    }

    fig, ax = plt.subplots(figsize=figsize)

    for mode in x["mode"].dropna().unique():
        s = x[x["mode"] == mode].sort_values("n_q")
        ax.plot(
            s["n_q"],
            s["median"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=mode,
            color=color_map.get(mode, None),
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, loc="left")
    ax.set_xlabel("Haltedauer (Quartale)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Median Ø Quartalsrendite (geom.)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(True, alpha=0.25)
    ax.set_xlim(n_min, n_max)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.set_xticks(list(range(n_min, n_max + 1, 4)))
    plt.tight_layout()
    return fig


def plot_triangle_matplotlib(
    tri: pd.DataFrame,
    title: str = "Rendite-Dreieck (Dynamik)",
    green_from: float = 0.0,
    figsize: tuple = (32, 18),
):
    if tri is None or tri.empty:
        return None

    import numpy as np
    import matplotlib.colors as mcolors
    from src.portfolio_simulation import quarter_to_index

    def _key(q):
        idx = quarter_to_index(q)
        return idx if idx is not None else 10**9

    buy_qs = sorted(tri["buy_q"].unique(), key=_key, reverse=True)
    sell_qs = sorted(tri["sell_q"].unique(), key=_key, reverse=False)

    pivot = tri.pivot(index="buy_q", columns="sell_q", values="value")
    pivot = pivot.reindex(index=buy_qs, columns=sell_qs)

    vals = pd.to_numeric(tri["value"], errors="coerce").dropna()
    if vals.empty:
        return None
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    m = max(abs(lo), abs(hi), 0.10)
    vmin, vmax = -m, m

    # redyellowgreen: exakt wie Altair-Skala
    # Rot bei vmin, Gelb bei green_from, Grün bei vmax
    pos_green = (green_from - vmin) / (vmax - vmin)
    pos_green = max(0.01, min(0.99, pos_green))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ryg",
        [
            (0.0,        "#A50026"),
            (pos_green * 0.25, "#D73027"),
            (pos_green * 0.55, "#FDAE61"),
            (pos_green * 0.85, "#FEE08B"),
            (pos_green,        "#FFFFBF"),
            (pos_green + (1 - pos_green) * 0.20, "#D9EF8B"),
            (pos_green + (1 - pos_green) * 0.50, "#91CF60"),
            (pos_green + (1 - pos_green) * 0.75, "#1A9850"),
            (1.0,        "#006837"),
        ],
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pivot.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    n_rows, n_cols = pivot.shape
    fontsize = max(14, min(16, int(400 / max(n_rows, n_cols))))

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.1%}",
                    ha="center", va="center",
                    fontsize=fontsize,
                    color="black",
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sell_qs, rotation=90, fontsize=14)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(buy_qs, fontsize=14)
    ax.set_xlabel("Verkauf (Quartal)", fontsize=14)
    ax.set_ylabel("Kauf (Quartal)", fontsize=14)
    ax.set_title(title, fontsize=14, loc="left")

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.005)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Ø Quartalsrendite (geom.)", fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.12)
    return fig


def plot_tax_effect_matplotlib(
    med_all: pd.DataFrame,
    title: str,
    n_min: int,
    n_max: int,
    figsize: tuple = (16, 5),
):
    if med_all is None or med_all.empty:
        return None

    x = med_all.copy()
    x["n_q"] = pd.to_numeric(x["n_q"], errors="coerce")
    x["median"] = pd.to_numeric(x["median"], errors="coerce")
    x = x.dropna(subset=["n_q", "median"]).copy()
    x["n_q"] = x["n_q"].astype(int)
    x = x[(x["n_q"] >= n_min) & (x["n_q"] <= n_max)].copy()

    if x.empty:
        return None

    color_map = {
        "Dynamik": "#1565C0",
        "Hold": "#66BB6A",
        "GSPC": "#EF5350",
        "SPXEW": "#F9A825",
    }

    fig, ax = plt.subplots(figsize=figsize)

    for mode in x["mode"].dropna().unique():
        s = x[x["mode"] == mode].sort_values("n_q")
        ax.plot(
            s["n_q"],
            s["median"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=mode,
            color=color_map.get(mode, None),
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, loc="left")
    ax.set_xlabel("Haltedauer (Quartale)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Steuereffekt: Brutto - Netto", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(True, alpha=0.25)
    ax.set_xlim(n_min, n_max)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.set_xticks(list(range(n_min, n_max + 1, 4)))
    plt.tight_layout()
    return fig


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Basis – Settings")
    q_start = st.text_input("q_start (YYYY-Qx)", value="2010-Q1")
    q_end = st.text_input("q_end (YYYY-Qx)", value="2025-Q4")
    max_n = st.number_input("max_n", min_value=1, max_value=120, value=60, step=1)
    end_cutoff = pd.to_datetime(st.date_input("end_cutoff", value=END_DATE_DEFAULT.date()))
    base_capital = st.number_input("base_capital", min_value=1.0, value=1000.0, step=100.0)

    st.divider()
    st.header("Steuern")
    tax_rate_saved = st.number_input("tax_rate Hold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    dyn_tax = st.number_input("tax_rate Dynamik", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

    st.divider()
    st.header("Benchmarks")
    bench_from = st.date_input("Benchmark von", value=date(2010, 1, 1))
    bench_to = st.date_input("Benchmark bis", value=date.today())
    tax_rate_bench = st.number_input("tax_rate Benchmarks", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

    st.divider()
    st.header("Dynamik")
    dyn_initial = st.number_input("Startkapital", min_value=0.0, value=1000.0, step=100.0)
    dyn_year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
    dyn_year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)

    st.divider()
    n_min = st.number_input("n_min", min_value=MIN_NQ, max_value=120, value=31, step=1)
    n_max = st.number_input("n_max", min_value=MIN_NQ, max_value=120, value=60, step=1)

download_figures = []

# ============================================================
# Daten bauen
# ============================================================
tri_saved_net = build_saved_triangle(
    q_start,
    q_end,
    end_cutoff,
    float(base_capital),
    float(tax_rate_saved),
    True,
)
tri_saved_gross = build_saved_triangle(
    q_start,
    q_end,
    end_cutoff,
    float(base_capital),
    0.0,
    False,
)

d_saved_net = extract_diagonals(tri_saved_net, int(max_n)) if not tri_saved_net.empty else pd.DataFrame()
d_saved_gross = extract_diagonals(tri_saved_gross, int(max_n)) if not tri_saved_gross.empty else pd.DataFrame()

if HAS_DYNAMIC:
    tri_dyn_net = compute_dynamic_triangle(
        es,
        dyn_initial,
        float(dyn_tax),
        int(dyn_year_from),
        int(dyn_year_to),
    )
    tri_dyn_gross = compute_dynamic_triangle(
        es,
        dyn_initial,
        0.0,
        int(dyn_year_from),
        int(dyn_year_to),
    )
else:
    tri_dyn_net = pd.DataFrame()
    tri_dyn_gross = pd.DataFrame()

d_dyn_net = extract_diagonals(tri_dyn_net, int(max_n)) if not tri_dyn_net.empty else pd.DataFrame()
d_dyn_gross = extract_diagonals(tri_dyn_gross, int(max_n)) if not tri_dyn_gross.empty else pd.DataFrame()

df_gspc = load_gspc()
df_spxew = load_spxew()

tri_gspc_net = build_benchmark_triangle_endonly_tax(
    df_gspc,
    float(tax_rate_bench),
    True,
    b_from=bench_from,
    b_to=bench_to,
)
tri_gspc_gross = build_benchmark_triangle_endonly_tax(
    df_gspc,
    0.0,
    False,
    b_from=bench_from,
    b_to=bench_to,
)
tri_spxew_net = build_benchmark_triangle_endonly_tax(
    df_spxew,
    float(tax_rate_bench),
    True,
    b_from=bench_from,
    b_to=bench_to,
)
tri_spxew_gross = build_benchmark_triangle_endonly_tax(
    df_spxew,
    0.0,
    False,
    b_from=bench_from,
    b_to=bench_to,
)

d_gspc_net = extract_diagonals(tri_gspc_net, int(max_n)) if not tri_gspc_net.empty else pd.DataFrame()
d_gspc_gross = extract_diagonals(tri_gspc_gross, int(max_n)) if not tri_gspc_gross.empty else pd.DataFrame()
d_spxew_net = extract_diagonals(tri_spxew_net, int(max_n)) if not tri_spxew_net.empty else pd.DataFrame()
d_spxew_gross = extract_diagonals(tri_spxew_gross, int(max_n)) if not tri_spxew_gross.empty else pd.DataFrame()

# ============================================================
# Chart 0: Violinplot Brutto vs Netto (Dynamik vs Hold)
# ============================================================
st.header("Aggregierte Verteilung – Brutto vs Netto – Haltedauer")

violin_df = build_violin_df(
    d_dyn_gross=d_dyn_gross,
    d_dyn_net=d_dyn_net,
    d_hold_gross=d_saved_gross,
    d_hold_net=d_saved_net,
    n_min=int(n_min),
    n_max=int(n_max),
)

fig_violin = plot_violin_matplotlib(
    violin_df,
    f"Aggregierte Verteilung – Brutto vs Netto – Haltedauer {int(n_min)}–{int(n_max)}",
    figsize=(14, 8),
)

if fig_violin is not None:
    st.pyplot(fig_violin, use_container_width=True)
    download_figures.append(("violin_brutto_vs_netto_hold.png", fig_violin))

    violin_buffer = io.BytesIO()
    fig_violin.savefig(violin_buffer, format="png", dpi=300, bbox_inches="tight")
    violin_buffer.seek(0)

    st.download_button(
        label="Violinplot als PNG herunterladen",
        data=violin_buffer,
        file_name="violin_brutto_vs_netto_hold.png",
        mime="image/png",
    )

# ============================================================
# Chart 1: Medianrenditen brutto
# ============================================================
st.header("Median-Rendite über Haltedauer – Brutto (Matplotlib)")

med_frames_gross = []
if not d_dyn_gross.empty:
    med_frames_gross.append(median_by_nq(d_dyn_gross, "Dynamik Brutto (ohne Steuer)"))
if not d_gspc_gross.empty:
    med_frames_gross.append(median_by_nq(d_gspc_gross, "GSPC Brutto (ohne Steuer)"))
if not d_saved_gross.empty:
    med_frames_gross.append(median_by_nq(d_saved_gross, "Hold Brutto (ohne Steuer)"))
if not d_spxew_gross.empty:
    med_frames_gross.append(median_by_nq(d_spxew_gross, "SPXEW Brutto (ohne Steuer)"))

med_all_gross = pd.concat(med_frames_gross, ignore_index=True) if med_frames_gross else pd.DataFrame()

fig0 = plot_median_lines_matplotlib(
    med_all_gross,
    f"Median-Linien Brutto – Haltedauer {int(n_min)}–{int(n_max)}",
    int(n_min),
    int(n_max),
    figsize=(16, 6.5),
)
if fig0 is not None:
    st.pyplot(fig0, use_container_width=True)
    download_figures.append(("median_brutto.png", fig0))

# ============================================================
# Chart 2: Medianrenditen netto
# ============================================================
st.header("Median-Rendite über Haltedauer – Netto (Matplotlib)")

med_frames_net = []
if not d_dyn_net.empty:
    med_frames_net.append(median_by_nq(d_dyn_net, "Dynamik Netto (mit Steuer)"))
if not d_gspc_net.empty:
    med_frames_net.append(median_by_nq(d_gspc_net, "GSPC Netto (mit Steuer)"))
if not d_saved_net.empty:
    med_frames_net.append(median_by_nq(d_saved_net, "Hold Netto (mit Steuer)"))
if not d_spxew_net.empty:
    med_frames_net.append(median_by_nq(d_spxew_net, "SPXEW Netto (mit Steuer)"))

med_all_net = pd.concat(med_frames_net, ignore_index=True) if med_frames_net else pd.DataFrame()

fig1 = plot_median_lines_matplotlib(
    med_all_net,
    f"Median-Linien Netto – Haltedauer {int(n_min)}–{int(n_max)}",
    int(n_min),
    int(n_max),
    figsize=(16, 6.5),
)
if fig1 is not None:
    st.pyplot(fig1, use_container_width=True)
    download_figures.append(("median_netto.png", fig1))

# ============================================================
# Chart 3: Steuereffekt
# ============================================================
st.header("Steuereffekt nach Haltedauer (Matplotlib)")

tax_frames = []
if not d_dyn_gross.empty and not d_dyn_net.empty:
    tax_frames.append(build_tax_effect_df(d_dyn_gross, d_dyn_net, "Dynamik"))
if not d_saved_gross.empty and not d_saved_net.empty:
    tax_frames.append(build_tax_effect_df(d_saved_gross, d_saved_net, "Hold"))
if not d_gspc_gross.empty and not d_gspc_net.empty:
    tax_frames.append(build_tax_effect_df(d_gspc_gross, d_gspc_net, "GSPC"))
if not d_spxew_gross.empty and not d_spxew_net.empty:
    tax_frames.append(build_tax_effect_df(d_spxew_gross, d_spxew_net, "SPXEW"))

tax_all = pd.concat(tax_frames, ignore_index=True) if tax_frames else pd.DataFrame()

fig2 = plot_tax_effect_matplotlib(
    tax_all,
    f"Steuereffekt nach Haltedauer – Brutto minus Netto ({int(n_min)}–{int(n_max)})",
    int(n_min),
    int(n_max),
    figsize=(16, 5),
)
if fig2 is not None:
    st.pyplot(fig2, use_container_width=True)
    download_figures.append(("steuereffekt.png", fig2))

# ============================================================
# Chart 4: Rendite-Dreieck Dynamik Brutto – Ausschnitt 2018–2025
# ============================================================
st.header("Rendite-Dreieck – Dynamik Brutto (Ausschnitt 2018–2025)")

if not tri_dyn_gross.empty:
    from src.portfolio_simulation import quarter_to_index

    def _filter_triangle_period(tri: pd.DataFrame, from_year: int, to_year: int) -> pd.DataFrame:
        def _year(q: str) -> int:
            try:
                return int(str(q).split("-")[0])
            except Exception:
                return 0

        mask = tri["buy_q"].apply(lambda q: from_year <= _year(q) <= to_year) & \
               tri["sell_q"].apply(lambda q: from_year <= _year(q) <= to_year)
        return tri[mask].copy()

    tri_filtered = _filter_triangle_period(tri_dyn_gross, 2018, 2025)

    fig_tri = plot_triangle_matplotlib(
        tri_filtered,
        title="Rendite-Dreieck (Dynamik, Brutto) – Ausschnitt 2018–2025",
        figsize=(36, 22),
    )
    if fig_tri is not None:
        st.pyplot(fig_tri, use_container_width=True)
        download_figures.append(("rendite_dreieck_dynamik_brutto_2018_2025.png", fig_tri))

        tri_buffer = io.BytesIO()
        fig_tri.savefig(tri_buffer, format="png", dpi=300, bbox_inches="tight")
        tri_buffer.seek(0)

        st.download_button(
            label="Rendite-Dreieck (Ausschnitt) als PNG herunterladen",
            data=tri_buffer,
            file_name="rendite_dreieck_dynamik_brutto_2018_2025.png",
            mime="image/png",
        )
else:
    st.info("Keine Dreiecksdaten für Dynamik Brutto verfügbar.")

# ============================================================
# Download
# ============================================================
st.divider()
st.header("Diagramme herunterladen")

if download_figures:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, fig in download_figures:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            img_buffer.seek(0)
            zf.writestr(filename, img_buffer.read())

    zip_buffer.seek(0)

    st.download_button(
        label="Alle Diagramme als ZIP mit PNGs herunterladen",
        data=zip_buffer,
        file_name="charts_matplotlib_png.zip",
        mime="application/zip",
    )