# pages/99_Debug_LD_Stichtage.py
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Streamlit config (muss früh kommen)
# ------------------------------------------------------------
st.set_page_config(page_title="Debug: ld-Stichtage", layout="wide")
st.title("Debug: ld-Stichtage je Portfolio (aus stocks)")

# ------------------------------------------------------------
# Pfad-Setup (Pages → src importierbar machen)
# ------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.funktionen import get_es_connection, list_portfolios, load_portfolio  # noqa: E402
from src.portfolio_simulation import (  # noqa: E402
    parse_portfolio_name,
    portfolio_doc_to_amount_weights,
    audit_ld_from_stocks,
)

# ------------------------------------------------------------
# Config + ES
# ------------------------------------------------------------
STOCKS_INDEX = os.getenv("ELASTICSEARCH_STOCKS_INDEX", "stocks")
es = get_es_connection()

st.caption("Ermittlung ld: stocks(calendarYear, period) → pro Symbol max(date), dann global max.")

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def _to_naive_ts(x) -> Optional[pd.Timestamp]:
    if x is None or pd.isna(x):
        return None
    t = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.tz_convert(None)


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("Filter")
    year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
    year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)
    only_problems = st.checkbox("Nur Probleme anzeigen (no_ld / no_syms / missing_doc)", value=False)


# ------------------------------------------------------------
# Compute Table
# ------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=300)
def compute_ld_table(
    _es_client,
    stocks_index: str,
    year_from: int,
    year_to: int,
) -> pd.DataFrame:
    ports = list_portfolios(_es_client)
    rows: List[Dict[str, Any]] = []

    for p in ports:
        pid = p.get("id")
        name = (p.get("name", "") or "").strip()
        pq = parse_portfolio_name(name)
        if not pid or pq is None:
            continue

        y, q = pq
        if y < year_from or y > year_to:
            continue

        doc = load_portfolio(_es_client, pid)
        if not doc:
            rows.append({
                "portfolio": name,
                "id": pid,
                "year": y,
                "quarter": f"Q{q}",
                "n_syms": 0,
                "ld": None,
                "driver_symbols": None,
                "min_max_date": None,
                "max_max_date": None,
                "spread_days": None,
                "status": "missing_doc",
            })
            continue

        w = portfolio_doc_to_amount_weights(doc)
        syms = sorted(w.keys())
        if not syms:
            rows.append({
                "portfolio": name,
                "id": pid,
                "year": y,
                "quarter": f"Q{q}",
                "n_syms": 0,
                "ld": None,
                "driver_symbols": None,
                "min_max_date": None,
                "max_max_date": None,
                "spread_days": None,
                "status": "no_syms",
            })
            continue

        df_ld, ld_global = audit_ld_from_stocks(_es_client, stocks_index, syms, y, q)

        if df_ld is None or df_ld.empty or ld_global is None or pd.isna(ld_global):
            rows.append({
                "portfolio": name,
                "id": pid,
                "year": y,
                "quarter": f"Q{q}",
                "n_syms": len(syms),
                "ld": None,
                "driver_symbols": None,
                "min_max_date": None,
                "max_max_date": None,
                "spread_days": None,
                "status": "no_ld",
            })
            continue

        # normalize tz
        df_ld = df_ld.copy()
        df_ld["max_date"] = pd.to_datetime(df_ld["max_date"], errors="coerce", utc=True).dt.tz_convert(None)

        min_d = df_ld["max_date"].min()
        max_d = df_ld["max_date"].max()
        drivers = df_ld.loc[df_ld["max_date"] == max_d, "symbol"].astype(str).tolist()

        rows.append({
            "portfolio": name,
            "id": pid,
            "year": y,
            "quarter": f"Q{q}",
            "n_syms": len(syms),
            "ld": _to_naive_ts(ld_global),
            "driver_symbols": ", ".join(drivers[:20]) + (" ..." if len(drivers) > 20 else ""),
            "min_max_date": min_d,
            "max_max_date": max_d,
            "spread_days": int((max_d - min_d).days) if pd.notna(min_d) and pd.notna(max_d) else None,
            "status": "ok",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["year", "quarter", "portfolio"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Main view
# ------------------------------------------------------------
df = compute_ld_table(es, STOCKS_INDEX, int(year_from), int(year_to))
if df.empty:
    st.warning("Keine Portfolios im angegebenen Bereich gefunden (oder Name nicht YYYY-Qx).")
    st.stop()

df_show = df[df["status"] != "ok"].copy() if only_problems else df.copy()

st.subheader("Übersicht: ld je Portfolio")
st.dataframe(df_show, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------
# Single-Portfolio Audit
# ------------------------------------------------------------
st.subheader("Ein Portfolio prüfen (pro Symbol max(date) + ld_global)")

ports_ok = df[df["status"] == "ok"][["portfolio", "id"]].copy()
if ports_ok.empty:
    st.info("Keine gültigen Portfolios (status=ok) im Filterbereich.")
    st.stop()

sel_label = st.selectbox("Portfolio wählen", ports_ok["portfolio"].tolist(), index=0, key="portfolio_sel")
pid = ports_ok.loc[ports_ok["portfolio"] == sel_label, "id"].iloc[0]

doc = load_portfolio(es, pid)
pq = parse_portfolio_name(doc.get("name", "")) if doc else None
if not doc or pq is None:
    st.warning("Portfolio konnte nicht geladen werden oder Name nicht YYYY-Qx.")
    st.stop()

y, q = pq
w = portfolio_doc_to_amount_weights(doc)
syms = sorted(w.keys())

df_ld, ld_global = audit_ld_from_stocks(es, STOCKS_INDEX, syms, y, q)

if df_ld is None or df_ld.empty:
    st.warning("audit_ld_from_stocks lieferte keine Daten.")
    st.stop()

# --- Anzeige: pro Symbol max_date + global ld ---
df_ld = df_ld.copy()
df_ld["max_date"] = pd.to_datetime(df_ld["max_date"], errors="coerce", utc=True).dt.tz_convert(None)
df_ld = df_ld.sort_values("max_date", ascending=False)

ld_naive = _to_naive_ts(ld_global)

st.caption(f"Globales ld (max über alle Symbol-max_date): {ld_naive}")
st.dataframe(df_ld, use_container_width=True, hide_index=True)

check = df_ld["max_date"].max()
st.write("Python-Check (df_ld['max_date'].max()):", check)
st.write("Identisch zu ld_global?", bool(check == ld_naive))

# ------------------------------------------------------------
# Extra Debug: Rohdaten je Symbol (ALLE Treffer für symbol+year+period)
# ------------------------------------------------------------
st.divider()
st.subheader("Symbol-Rohdaten im Quarter (symbol + year + period)")

sym_sel = st.selectbox(
    "Symbol auswählen",
    df_ld["symbol"].astype(str).tolist(),
    index=0,
    key="sym_sel_quarter",
)

row_sel = df_ld.loc[df_ld["symbol"].astype(str) == str(sym_sel)].head(1)
if row_sel.empty or pd.isna(row_sel["max_date"].iloc[0]):
    st.warning("Für dieses Symbol gibt es kein max_date in df_ld.")
    st.stop()

max_dt = pd.Timestamp(row_sel["max_date"].iloc[0])

year_str = str(y)
period = f"Q{q}"

st.caption(f"Filter: symbol={sym_sel} | calendarYear={year_str} | period={period}")
st.caption(f"max_date aus Aggregation: {max_dt.strftime('%Y-%m-%d')}")

body = {
    "size": 5000,
    "_source": ["symbol", "date", "calendarYear", "period", "source", "ingested_at"],
    "query": {
        "bool": {
            "filter": [
                {"term": {"symbol": str(sym_sel)}},
                {"term": {"calendarYear.keyword": year_str}},
                {"term": {"period.keyword": period}},
            ]
        }
    },
    "sort": [{"date": "asc"}],
}

try:
    resp = es.search(index=STOCKS_INDEX, body=body)
    hits = resp.get("hits", {}).get("hits", [])
except Exception as e:
    hits = []
    st.error(f"ES Query Fehler: {e}")

if not hits:
    st.info("Keine Rohdaten für diesen Filter gefunden (prüfe Mapping / Quarter-Partition).")
    st.stop()

rows_raw = []
for h in hits:
    src = h.get("_source", {}) or {}
    rows_raw.append({
        "date": src.get("date"),
        "symbol": src.get("symbol"),
        "calendarYear": src.get("calendarYear"),
        "period": src.get("period"),
        "source": src.get("source"),
        "ingested_at": src.get("ingested_at"),
        "_id": h.get("_id"),
    })

df_raw = pd.DataFrame(rows_raw)
df_raw["date_ts"] = pd.to_datetime(df_raw["date"], errors="coerce")
df_raw["is_max_date"] = df_raw["date_ts"] == max_dt

st.dataframe(df_raw.drop(columns=["date_ts"]), use_container_width=True, hide_index=True)

st.write("Anzahl Treffer:", int(len(df_raw)))
st.write("Letztes Datum in Rohdaten:", df_raw["date_ts"].max())
st.write("Gibt es Einträge nach max_date?", bool((df_raw["date_ts"] > max_dt).any()))
# ------------------------------------------------------------
# Extra Debug: Symbol-Historie (Quarter-Dates 2010–heute)
# ------------------------------------------------------------
st.divider()
st.subheader("Symbol-Historie (Quarter-Dates 2010–heute, unabhängig vom Portfolio-Quarter)")

sym_hist = st.selectbox(
    "Symbol (Historie)",
    syms,  # nur Symbole aus dem ausgewählten Portfolio
    index=0,
    key="sym_sel_history",
)

min_year = st.number_input("ab calendarYear", min_value=1990, max_value=2100, value=2010, step=1, key="hist_min_year")
dedup = st.checkbox("Je (calendarYear, period) nur das letzte date (dedup)", value=True, key="hist_dedup")

body_hist = {
    "size": 5000,
    "_source": ["symbol", "calendarYear", "period", "date", "source", "ingested_at"],
    "query": {
        "bool": {
            "filter": [
                {"term": {"symbol": str(sym_hist)}},
                {"terms": {"period.keyword": ["Q1", "Q2", "Q3", "Q4"]}},
            ]
        }
    },
    "sort": [{"date": "asc"}],
}

try:
    resp_h = es.search(index=STOCKS_INDEX, body=body_hist)
    hits_h = resp_h.get("hits", {}).get("hits", [])
except Exception as e:
    hits_h = []
    st.error(f"ES Query Fehler (Historie): {e}")

if not hits_h:
    st.info("Keine Historie gefunden.")
else:
    rows_h = []
    for h in hits_h:
        src = h.get("_source", {}) or {}
        rows_h.append({
            "calendarYear": src.get("calendarYear"),
            "period": src.get("period"),
            "date": src.get("date"),
            "source": src.get("source"),
            "ingested_at": src.get("ingested_at"),
            "_id": h.get("_id"),
        })

    df_h = pd.DataFrame(rows_h)

    # Typen sauber machen
    df_h["calendarYear_int"] = pd.to_numeric(df_h["calendarYear"], errors="coerce")
    df_h["date_ts"] = pd.to_datetime(df_h["date"], errors="coerce")

    # ab Jahr filtern (weil calendarYear in ES text ist)
    df_h = df_h[df_h["calendarYear_int"].notna()]
    df_h = df_h[df_h["calendarYear_int"] >= int(min_year)].copy()

    # optional: je (calendarYear, period) den "letzten" Eintrag behalten
    if dedup and not df_h.empty:
        df_h = (
            df_h.sort_values(["calendarYear_int", "period", "date_ts"])
               .groupby(["calendarYear_int", "period"], as_index=False)
               .tail(1)
               .sort_values(["calendarYear_int", "period"])
               .reset_index(drop=True)
        )

    # Anzeige
    df_show_h = df_h.drop(columns=["calendarYear_int", "date_ts"], errors="ignore")
    st.dataframe(df_show_h, use_container_width=True, hide_index=True)

    # kleine Zusammenfassung
    if not df_h.empty:
        st.write("Anzahl Einträge:", int(len(df_h)))
        st.write("Jahr-Spanne:", int(df_h["calendarYear_int"].min()), "bis", int(df_h["calendarYear_int"].max()))
