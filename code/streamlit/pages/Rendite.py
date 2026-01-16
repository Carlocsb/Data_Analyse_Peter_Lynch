import os
import sys
import re
import pandas as pd
import streamlit as st
import altair as alt
from datetime import date
from typing import Any, Dict, List, Optional

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

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PRICE_INDEX = os.getenv("ELASTICSEARCH_PRICE_INDEX", "prices")
BENCH_INDEX = os.getenv("ELASTICSEARCH_BENCH_INDEX", "benchmarks")
STOCKS_INDEX = os.getenv("ELASTICSEARCH_STOCKS_INDEX", "stocks")
GSPC_SYMBOL = os.getenv("GSPC_SYMBOL", "^GSPC")

st.set_page_config(page_title="Rendite-Dreieck & S&P 500", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png")
st.title("Rendite-Dreieck (Heatmap) & S&P 500 (Linie)")

es = get_es_connection()

# ------------------------------------------------------------
# ES Helper
# ------------------------------------------------------------
def es_fetch_all_by_symbol(index: str, symbol: str, source_fields: List[str]) -> pd.DataFrame:
    """Pagination via search_after; erwartet sort= date asc."""
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


# ------------------------------------------------------------
# Rendite-Dreieck (Jahr → Jahr, CAGR p.a.)
# ------------------------------------------------------------
def cagr(p0: float, p1: float, years: float) -> Optional[float]:
    if p0 <= 0 or p1 <= 0 or years <= 0:
        return None
    return (p1 / p0) ** (1.0 / years) - 1.0


def build_return_triangle(df: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Erwartet df mit Spalten: date, adjClose (bei Portfolio = Portfolio-Wert).
    Nimmt letzten Handelstag je Jahr als Jahresendstand, berechnet CAGR p.a.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["year"] = df["date"].dt.year

    eoy = df.sort_values("date").groupby("year", as_index=False).tail(1)[["year", "date", "adjClose"]]
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


# ------------------------------------------------------------
# Portfolio → synthetische Wertreihe (Buy & Hold, täglich)
# ------------------------------------------------------------
def portfolio_doc_to_amounts(portfolio_doc: Dict[str, Any]) -> Dict[str, float]:
    amounts: Dict[str, float] = {}
    for it in portfolio_doc.get("items", []) or []:
        sym = it.get("symbol")
        try:
            amt = float(it.get("amount", 0) or 0)
        except Exception:
            amt = 0.0
        if sym and amt > 0:
            amounts[sym] = amounts.get(sym, 0.0) + amt
    return amounts


def build_portfolio_series(
    symbols_amounts: Dict[str, float],
    year_from: int,
    year_to: int,
    ffill_missing: bool = True,
) -> pd.DataFrame:
    """
    Buy & Hold (täglich):
      shares_i = amount_i / price_i(t0)
      portfolio_value(t) = Σ shares_i * price_i(t)
    """
    invest_syms = [s for s, a in symbols_amounts.items() if a and a > 0]
    if not invest_syms:
        return pd.DataFrame()

    frames = []
    for sym in invest_syms:
        df = load_prices(sym)
        if df.empty:
            continue

        df = df.copy()
        df["year"] = df["date"].dt.year
        df = df[(df["year"] >= year_from) & (df["year"] <= year_to)]
        if df.empty:
            continue

        frames.append(df[["date", "adjClose"]].rename(columns={"adjClose": sym}).set_index("date"))

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).sort_index()

    if ffill_missing:
        prices = prices.ffill()

    if not set(invest_syms).issubset(set(prices.columns)):
        missing = sorted(list(set(invest_syms) - set(prices.columns)))
        st.warning(f"Für folgende Symbole fehlen Preisdaten: {', '.join(missing)}")
        invest_syms = [s for s in invest_syms if s in prices.columns]

    prices_sub = prices[invest_syms].dropna(how="any")
    if prices_sub.empty:
        return pd.DataFrame()

    t0 = prices_sub.index[0]
    p0 = prices.loc[t0, invest_syms]

    shares = {}
    for sym in invest_syms:
        amt = float(symbols_amounts.get(sym, 0.0))
        px0 = float(p0[sym]) if pd.notna(p0[sym]) else 0.0
        if amt > 0 and px0 > 0:
            shares[sym] = amt / px0

    if not shares:
        return pd.DataFrame()

    used_syms = list(shares.keys())
    port_val = (prices[used_syms] * pd.Series(shares)).sum(axis=1)

    out = port_val.dropna().reset_index()
    out.columns = ["date", "adjClose"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["adjClose"] = pd.to_numeric(out["adjClose"], errors="coerce")
    out = out.dropna().sort_values("date")
    return out


# ------------------------------------------------------------
# Dynamischer Algo (Quartals-Umschichtung)
#   ld aus stocks: pro Symbol max(date) bei calendarYear+period, dann global max
#   Buy weights aus Portfolio: w_i = amount_i / sum(amount)
#   Prices: adjClose am/ vor ld
# ------------------------------------------------------------
QRE = re.compile(r"^(?P<y>\d{4})-Q(?P<q>[1-4])$")


def parse_portfolio_name(name: str) -> Optional[tuple[int, int]]:
    if not name:
        return None
    m = QRE.match(str(name).strip())
    if not m:
        return None
    return int(m.group("y")), int(m.group("q"))


def sort_portfolios_quarterly(portfolios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tmp = []
    for p in portfolios:
        k = parse_portfolio_name(p.get("name", ""))
        if k:
            tmp.append((k, p))
    tmp.sort(key=lambda x: x[0])
    return [p for _, p in tmp]


def portfolio_doc_to_amount_weights(portfolio_doc: Dict[str, Any]) -> Dict[str, float]:
    amounts = portfolio_doc_to_amounts(portfolio_doc)
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {sym: amt / total for sym, amt in amounts.items()}


def ld_from_stocks(es, symbols: List[str], year: int, quarter: int) -> Optional[pd.Timestamp]:
    """pro Symbol max(date) in stocks (calendarYear, period), dann global max."""
    if not symbols:
        return None
    period = f"Q{quarter}"

    for sym_field in ["symbol.keyword", "symbol"]:
        for per_field in ["period.keyword", "period"]:
            body = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {sym_field: symbols}},
                            {"term": {"calendarYear": year}},
                            {"term": {per_field: period}},
                        ]
                    }
                },
                "aggs": {
                    "by_symbol": {
                        "terms": {"field": sym_field, "size": len(symbols)},
                        "aggs": {"max_date": {"max": {"field": "date"}}},
                    }
                },
            }
            try:
                resp = es.search(index=STOCKS_INDEX, body=body)
            except Exception:
                continue

            buckets = resp.get("aggregations", {}).get("by_symbol", {}).get("buckets", [])
            ds: List[pd.Timestamp] = []
            for b in buckets:
                agg = b.get("max_date", {})
                s = agg.get("value_as_string")
                v = agg.get("value")
                if s:
                    d = pd.to_datetime(s, utc=True, errors="coerce")
                elif v is not None:
                    d = pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
                else:
                    d = pd.NaT
                if pd.notna(d):
                    ds.append(d)

            if ds:
                return max(ds)

    return None


def price_on_or_before(es, symbol: str, ld: pd.Timestamp) -> Optional[float]:
    ld = pd.to_datetime(ld, utc=True, errors="coerce")
    if pd.isna(ld):
        return None
    ld_str = ld.strftime("%Y-%m-%d")  # prices speichern YYYY-MM-DD

    for sym_field in ["symbol.keyword", "symbol"]:
        body = {
            "size": 1,
            "_source": ["symbol", "date", "adjClose"],
            "query": {
                "bool": {
                    "filter": [
                        {"term": {sym_field: symbol}},
                        {"range": {"date": {"lte": ld_str}}},
                    ]
                }
            },
            "sort": [{"date": "desc"}],
        }
        try:
            resp = es.search(index=PRICE_INDEX, body=body)
        except Exception:
            continue

        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            continue
        src = hits[0].get("_source", {})
        try:
            return float(src.get("adjClose"))
        except Exception:
            return None

    return None


@st.cache_data(show_spinner=True, ttl=300)
def simulate_dynamic_cached(
    portfolio_minimal: List[Dict[str, str]],
    initial_capital: float,
    tax_rate: float,
    year_from: int,
    year_to: int,
) -> pd.DataFrame:
    """
    Cache-Wrapper: nimmt nur (id,name) als Input.
    Lädt Portfolio-Docs innerhalb der Funktion über ES (global: es).
    """
    ports_full = [{"id": p["id"], "name": p["name"]} for p in portfolio_minimal]
    ports = sort_portfolios_quarterly(ports_full)
    ports = [p for p in ports if (parse_portfolio_name(p.get("name", "")) is not None)]
    ports = [p for p in ports if year_from <= parse_portfolio_name(p["name"])[0] <= year_to]

    if len(ports) < 2:
        return pd.DataFrame()

    holdings: Dict[str, Dict[str, float]] = {}  # sym -> {"shares", "buy_price"}
    nav_prev = float(initial_capital)
    results = []

    for idx in range(1, len(ports)):
        neu_p = ports[idx]

        neu_name = neu_p.get("name", "")
        pq = parse_portfolio_name(neu_name)
        if pq is None:
            continue
        year, quarter = pq

        neu_doc = load_portfolio(es, neu_p.get("id"))
        if not neu_doc:
            continue

        weights = portfolio_doc_to_amount_weights(neu_doc)
        new_syms = sorted(weights.keys())
        if not new_syms:
            continue

        ld = ld_from_stocks(es, new_syms, year, quarter)
        if ld is None or pd.isna(ld):
            continue

        gross_value = 0.0
        tax = 0.0

        if holdings:
            for sym, pos in holdings.items():
                px = price_on_or_before(es, sym, ld)
                if px is None:
                    continue
                shares = float(pos.get("shares", 0.0))
                buy_px = float(pos.get("buy_price", 0.0))

                proceeds = shares * px
                pnl = (px - buy_px) * shares

                gross_value += proceeds
                if pnl > 0:
                    tax += float(tax_rate) * pnl
        else:
            gross_value = nav_prev
            tax = 0.0

        end_value = gross_value - tax
        ret_q = (end_value / nav_prev) - 1.0 if nav_prev > 0 else float("nan")

        # Neu komplett kaufen (amount-gewichtet)
        new_holdings: Dict[str, Dict[str, float]] = {}
        for sym, w in weights.items():
            px = price_on_or_before(es, sym, ld)
            if px is None or px <= 0:
                continue
            alloc = end_value * float(w)
            shares = alloc / px
            new_holdings[sym] = {"shares": shares, "buy_price": px}

        results.append(
            {
                "quarter": neu_name,
                "ld": pd.to_datetime(ld).tz_convert(None),  # naive für Charts/Tabellen
                "start_value": nav_prev,
                "gross_value": gross_value,
                "tax": tax,
                "end_value": end_value,
                "return_q": ret_q,
                "n_old": len(holdings),
                "n_new": len(new_holdings),
            }
        )

        holdings = new_holdings
        nav_prev = end_value

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["ld"] = pd.to_datetime(df["ld"], errors="coerce")
    for c in ["start_value", "gross_value", "tax", "end_value", "return_q"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ld", "end_value"]).sort_values("ld")
    return df


# ------------------------------------------------------------
# Quartals-Dreieck (simple oder CAGR p.a. pro Zelle)
# ------------------------------------------------------------
def build_quarter_triangle(df_dyn: pd.DataFrame, metric: str = "simple") -> pd.DataFrame:
    """
    df_dyn Spalten: quarter (str), ld (datetime), end_value (float)

    metric:
      - "simple": kumulierte Rendite (NAV_sell / NAV_buy - 1)
      - "cagr":   CAGR p.a. über echte Zeit (ld_sell - ld_buy)
    """
    if df_dyn is None or df_dyn.empty:
        return pd.DataFrame()

    df = df_dyn.copy()
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

            nav_buy = nav[i]
            nav_sell = nav[j]
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


def plot_quarter_triangle(tri: pd.DataFrame, metric: str = "simple") -> alt.Chart:
    title = "Rendite" if metric == "simple" else "CAGR p.a."
    fmt_tooltip = ".2%"
    fmt_text = ".1%"

    # gewünschte Reihenfolge: newest -> oldest
    sell_order = sorted(tri["sell_q"].unique(), reverse=True)
    buy_order  = sorted(tri["buy_q"].unique(), reverse=True)

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



# ------------------------------------------------------------
# UI (Sidebar)
# ------------------------------------------------------------
with st.sidebar:
    st.header("Einstellungen")

    mode = st.radio(
        "Rendite-Dreieck Quelle",
        ["Einzeltitel", "Gespeichertes Portfolio", "Dynamisch (Quartals-Umstieg)"],
        index=0,
    )

    symbols = get_symbols_from_prices()
    symbol = None
    portfolio_doc = None
    portfolio_amounts: Dict[str, float] = {}

    if mode == "Einzeltitel":
        if not symbols:
            st.warning(f"Keine Symbole im Index '{PRICE_INDEX}' gefunden.")
        else:
            default_sym = "AAPL" if "AAPL" in symbols else symbols[0]
            symbol = st.selectbox("Aktie (Rendite-Dreieck)", symbols, index=symbols.index(default_sym))

    elif mode == "Gespeichertes Portfolio":
        portfolios = list_portfolios(es)
        if not portfolios:
            st.warning("Keine gespeicherten Portfolios gefunden.")
        else:
            labels = ["—"] + [f'{p.get("name","(ohne Name)")} — {p.get("id")}' for p in portfolios]
            sel = st.selectbox("Portfolio (Rendite-Dreieck)", labels, index=0)
            if sel != "—":
                idx = labels.index(sel) - 1
                pid = portfolios[idx].get("id")
                if pid:
                    portfolio_doc = load_portfolio(es, pid)
                    if portfolio_doc:
                        portfolio_amounts = portfolio_doc_to_amounts(portfolio_doc)
                        st.caption("Portfolio-Symbole & Beträge")
                        df_p = pd.DataFrame(
                            [{"Symbol": k, "Amount (USD)": v} for k, v in sorted(portfolio_amounts.items())]
                        )
                        st.dataframe(df_p, use_container_width=True, hide_index=True)

    else:
        st.caption("Dynamik: nutzt alle Quartals-Portfolios aus ES (YYYY-Qx).")

    st.divider()
    st.caption("Investment (Dynamik / Benchmarks)")
    initial_investment = st.number_input(
        "Startkapital (USD)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )

    tax_rate = 0.2
    show_debug_table = True
    show_charts = True

    if mode == "Dynamisch (Quartals-Umstieg)":
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
    else:
        triangle_view = "Jahre (CAGR p.a.)"

    st.divider()
    st.caption("Zeitraum (für Dreieck & Dynamik)")
    year_from = st.number_input("Von Jahr", min_value=1990, max_value=2100, value=2010, step=1)
    year_to = st.number_input("Bis Jahr", min_value=1990, max_value=2100, value=2025, step=1)

    st.divider()
    show_sp = st.checkbox("S&P 500 (GSPC) anzeigen", value=True)
    if show_sp:
        with st.expander("S&P 500 Zeitraum", expanded=False):
            gspc_from = st.date_input("Von Datum", value=date(2010, 1, 1))
            gspc_to = st.date_input("Bis Datum", value=date.today())
    else:
        gspc_from = date(2010, 1, 1)
        gspc_to = date.today()


# ------------------------------------------------------------
# Main: Rendite-Dreieck
# ------------------------------------------------------------
st.subheader("Rendite-Dreieck")

df_base = pd.DataFrame()
df_dyn = pd.DataFrame()

if mode == "Einzeltitel":
    if symbol is None:
        st.stop()
    df_base = load_prices(symbol)
    if df_base.empty:
        st.warning("Keine Preisdaten gefunden.")
        st.stop()

elif mode == "Gespeichertes Portfolio":
    if not portfolio_doc:
        st.info("Bitte ein gespeichertes Portfolio auswählen.")
        st.stop()
    if not portfolio_amounts:
        st.warning("Portfolio enthält keine Beträge > 0 oder keine Items.")
        st.stop()

    df_base = build_portfolio_series(
        symbols_amounts=portfolio_amounts,
        year_from=int(year_from),
        year_to=int(year_to),
        ffill_missing=True,
    )
    if df_base.empty:
        st.warning("Portfolio-Zeitreihe konnte nicht berechnet werden (fehlende Daten / Zeitraum).")
        st.stop()

else:
    portfolios = list_portfolios(es)
    if not portfolios:
        st.warning("Keine gespeicherten Portfolios gefunden.")
        st.stop()

    minimal = [{"id": p.get("id"), "name": p.get("name", "")} for p in portfolios if p.get("id")]
    df_dyn = simulate_dynamic_cached(
        portfolio_minimal=minimal,
        initial_capital=float(initial_investment),
        tax_rate=float(tax_rate),
        year_from=int(year_from),
        year_to=int(year_to),
    )
    if df_dyn.empty:
        st.warning("Keine dynamischen Ergebnisse (prüfe stocks/prices Daten oder Zeitraum).")
        st.stop()

    # Für Jahres-Dreieck bauen wir eine "Pseudo-Preisreihe" aus NAVs
    df_base = df_dyn[["ld", "end_value"]].rename(columns={"ld": "date", "end_value": "adjClose"}).copy()

# ---- QUARTALS-DREIECK (nur wenn Dynamik + Quartal gewählt)
if mode == "Dynamisch (Quartals-Umstieg)" and triangle_view.startswith("Quartale"):
    metric = "cagr" if "CAGR" in triangle_view else "simple"
    tri_q = build_quarter_triangle(df_dyn, metric=metric)

    if tri_q.empty:
        st.warning("Quartals-Rendite-Dreieck konnte nicht berechnet werden.")
    else:
        st.altair_chart(plot_quarter_triangle(tri_q, metric=metric), use_container_width=True)
        if metric == "simple":
            st.caption("Quartals-Dreieck: Zelle = kumulierte Rendite (NAV_sell / NAV_buy - 1).")
        else:
            st.caption("Quartals-Dreieck: Zelle = CAGR p.a. (annualisiert über ld-Differenz).")

# ---- JAHRES-DREIECK (default)
else:
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

            if mode == "Einzeltitel":
                st.caption("CAGR p.a. basierend auf Jahresendständen (letzter Handelstag je Jahr).")
            elif mode == "Gespeichertes Portfolio":
                st.caption(
                    "Portfolio: Buy-&-Hold (Beträge → Shares am t0), "
                    "CAGR p.a. basierend auf Portfolio-Wert (EOY, letzter Handelstag je Jahr)."
                )
            else:
                st.caption(
                    "Dynamisch: Quartals-Umschichtung; ld aus stocks (calendarYear+period, "
                    "pro Symbol max(date) → global max); NAV = End Value pro Quartal."
                )

# ------------------------------------------------------------
# Main: Dynamische Strategie – Ergebnisse (unter dem Dreieck)
# ------------------------------------------------------------
st.subheader("Dynamische Strategie – Ergebnisse")

if mode != "Dynamisch (Quartals-Umstieg)":
    st.info("Wähle links den Modus „Dynamisch (Quartals-Umstieg)“, um KPIs/Charts zu sehen.")
else:
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
                tooltip=[
                    alt.Tooltip("quarter:N", title="Quartal"),
                    alt.Tooltip("tax:Q", title="Steuer", format=",.2f"),
                ],
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
        show_cols = ["quarter", "ld", "start_value", "gross_value", "tax", "end_value", "return_q", "n_old", "n_new"]
        df_show = df_dyn[show_cols].copy()
        df_show["return_q"] = df_show["return_q"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Main: S&P 500 (unter dem Dreieck, einklappbar)
# ------------------------------------------------------------
if show_sp:
    with st.expander("S&P 500 (GSPC) – Linie", expanded=False):
        df_g = load_gspc()
        if df_g.empty:
            st.warning(f"Keine Daten für {GSPC_SYMBOL} im Index '{BENCH_INDEX}' gefunden.")
        else:
            mask = (df_g["date"].dt.date >= gspc_from) & (df_g["date"].dt.date <= gspc_to)
            g = df_g.loc[mask].copy()
            if g.empty:
                st.warning("Keine Daten im gewählten Zeitraum.")
            else:
                line = (
                    alt.Chart(g)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Datum"),
                        y=alt.Y("adjClose:Q", title="adjClose"),
                        tooltip=[
                            alt.Tooltip("date:T", title="Datum"),
                            alt.Tooltip("adjClose:Q", title="adjClose", format=",.2f"),
                        ],
                    )
                    .properties(height=420)
                )
                st.altair_chart(line, use_container_width=True)

                start = float(g.iloc[0]["adjClose"])
                end = float(g.iloc[-1]["adjClose"])

                final_value = initial_investment * (end / start) if start > 0 else float("nan")
                total_return_sp = (end / start) - 1.0 if start > 0 else float("nan")

                yrs = (g["date"].iloc[-1] - g["date"].iloc[0]).days / 365.25
                cagr_val_sp = (end / start) ** (1.0 / yrs) - 1.0 if start > 0 and yrs > 0 else float("nan")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Gesamtrendite", f"{total_return_sp*100:.1f}%")
                m2.metric("CAGR p.a.", f"{cagr_val_sp*100:.1f}%")
                m3.metric("Startkapital", f"{initial_investment:,.0f} USD")
                m4.metric("Endwert", f"{final_value:,.0f} USD")
plot_quarter_triangle