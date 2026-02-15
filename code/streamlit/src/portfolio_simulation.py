# src/portfolio_simulation.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ============================================================
# Naming / Quarter parsing
# ============================================================
_QRE = re.compile(r"^(?P<y>\d{4})-Q(?P<q>[1-4])$")


def parse_portfolio_name(name: str) -> Optional[tuple[int, int]]:
    if not name:
        return None
    m = _QRE.match(str(name).strip())
    if not m:
        return None
    return int(m.group("y")), int(m.group("q"))


# ============================================================
# Portfolio weights
# ============================================================
def portfolio_doc_to_amounts(portfolio_doc: Dict[str, Any]) -> Dict[str, float]:
    """Summiert Amount pro Symbol (falls Symbol mehrfach in items vorkommt)."""
    amounts: Dict[str, float] = {}
    for it in (portfolio_doc or {}).get("items", []) or []:
        sym = it.get("symbol")
        try:
            amt = float(it.get("amount", 0) or 0)
        except Exception:
            amt = 0.0
        if sym and amt > 0:
            s = str(sym).strip()
            amounts[s] = amounts.get(s, 0.0) + amt
    return amounts


def portfolio_doc_to_amount_weights(portfolio_doc: Dict[str, Any]) -> Dict[str, float]:
    amounts = portfolio_doc_to_amounts(portfolio_doc)
    total = float(sum(amounts.values()))
    if total <= 0:
        return {}
    return {sym: float(amt) / total for sym, amt in amounts.items()}


# ============================================================
# LD from stocks (for dynamic + saved buy date)
# ============================================================
def ld_from_stocks(
    es_client,
    stocks_index: str,
    symbols: List[str],
    year: int,
    quarter: int,
) -> Optional[pd.Timestamp]:
    """
    Pro Symbol max(date) in stocks (calendarYear, period), dann global max.
    Erwartete Mapping-Felder:
      - symbol: keyword -> "symbol"
      - calendarYear: text+keyword -> "calendarYear.keyword"
      - period: text+keyword -> "period.keyword"
    """
    if not symbols:
        return None

    period = f"Q{quarter}"

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"symbol": symbols}},
                    {"term": {"calendarYear.keyword": str(year)}},
                    {"term": {"period.keyword": period}},
                ]
            }
        },
        "aggs": {
            "by_symbol": {
                "terms": {"field": "symbol", "size": len(symbols)},
                "aggs": {"max_date": {"max": {"field": "date"}}},
            }
        },
    }

    resp = es_client.search(index=stocks_index, body=body)
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

    return max(ds) if ds else None


def audit_ld_from_stocks(
    es_client,
    stocks_index: str,
    symbols: List[str],
    year: int,
    quarter: int,
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """
    Debug/Audit: liefert pro Symbol das max_date + raw value_as_string/value,
    plus globales ld.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "max_date", "max_date_raw"]), None

    period = f"Q{quarter}"

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"symbol": symbols}},
                    {"term": {"calendarYear.keyword": str(year)}},
                    {"term": {"period.keyword": period}},
                ]
            }
        },
        "aggs": {
            "by_symbol": {
                "terms": {"field": "symbol", "size": len(symbols)},
                "aggs": {"max_date": {"max": {"field": "date"}}},
            }
        },
    }

    resp = es_client.search(index=stocks_index, body=body)
    buckets = resp.get("aggregations", {}).get("by_symbol", {}).get("buckets", [])

    rows: List[Dict[str, Any]] = []
    for b in buckets:
        sym = b.get("key")
        agg = b.get("max_date", {})
        s = agg.get("value_as_string")
        v = agg.get("value")

        if s:
            d = pd.to_datetime(s, utc=True, errors="coerce")
        elif v is not None:
            d = pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
        else:
            d = pd.NaT

        rows.append({"symbol": sym, "max_date": d, "max_date_raw": s or v})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, None

    df = df.sort_values("max_date", ascending=False).reset_index(drop=True)
    ld_global = df["max_date"].dropna().max() if df["max_date"].notna().any() else None
    return df, ld_global


def get_buy_date_like_dynamic_for_portfolio(
    es_client,
    stocks_index: str,
    portfolio_doc: Dict[str, Any],
) -> Optional[pd.Timestamp]:
    """
    Ableitung Kaufdatum wie Dynamik:
    - Portfolio-Name muss YYYY-Qx sein
    - Gewichte aus items
    - ld_from_stocks(...) liefert den ld-Stichtag
    """
    name = (portfolio_doc or {}).get("name", "")
    pq = parse_portfolio_name(name)
    if pq is None:
        return None
    y, q = pq

    w = portfolio_doc_to_amount_weights(portfolio_doc)
    syms = sorted(w.keys())
    if not syms:
        return None

    ld = ld_from_stocks(es_client, stocks_index, syms, y, q)
    if ld is None or pd.isna(ld):
        return None

    # -> naiv (local) zurückgeben
    return pd.to_datetime(ld).tz_convert(None)


# ============================================================
# Buy&Hold with liquidation per symbol (Saved portfolios)
# ============================================================
def next_trading_day(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(pd.Timestamp(dt), side="left")
    if pos >= len(index):
        return None
    return index[pos]


def last_valid_price_date(series: pd.Series, end_dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    s = series.loc[:end_dt].dropna()
    if s.empty:
        return None
    return s.index[-1]


def build_saved_buyhold_series_with_liquidation(
    prices: pd.DataFrame,  # index=date, columns=symbols (adjClose)
    weights: Dict[str, float],  # sum ~ 1
    buy_date: pd.Timestamp,  # ld wie dynamisch
    initial_capital: float,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simuliert Buy-&-Hold ab buy_date (gemappt auf nächsten Handelstag im prices-Index),
    Liquidation je Symbol:
    - bevorzugt end_date, sonst letzter Preis <= end_date.
    - nach Liquidation: Position -> Cash (Cash bleibt im NAV).
    Gibt zurück:
      df_nav:    columns [date, nav, cash]
      df_trades: BUY/SELL/SKIP Log
    """
    if prices is None or prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    px_all = prices.sort_index().copy()
    idx_all = px_all.index

    buy_dt = next_trading_day(idx_all, pd.Timestamp(buy_date))
    if buy_dt is None:
        return pd.DataFrame(), pd.DataFrame()

    # reporting window endet am letzten global verfügbaren Tag <= end_date
    if (idx_all <= end_date).any():
        report_end = idx_all[idx_all <= end_date][-1]
    else:
        report_end = idx_all[-1]

    px = px_all.loc[(px_all.index >= buy_dt) & (px_all.index <= report_end)]
    if px.empty:
        return pd.DataFrame(), pd.DataFrame()

    # weights normalisieren & nur Symbole mit Spalte
    w = pd.Series({k: float(v) for k, v in (weights or {}).items() if k in px_all.columns and v and v > 0})
    if w.empty:
        return pd.DataFrame(), pd.DataFrame()
    w = w / w.sum()

    # Kaufpreise am buy_dt (ohne ffill)
    px_buy = px_all.loc[buy_dt, w.index]
    valid = px_buy.dropna()
    invalid_syms = sorted(set(w.index) - set(valid.index))

    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    # remove invalid from weights, renormalize
    w = w.loc[valid.index]
    w = w / w.sum()

    shares = (initial_capital * w / valid).astype(float)

    # Liquidation je Symbol
    sell_dt: Dict[str, pd.Timestamp] = {}
    sell_px: Dict[str, float] = {}
    reason: Dict[str, str] = {}

    for sym in w.index:
        dt_last = last_valid_price_date(px_all[sym], end_date)
        if dt_last is None:
            # kein Kurs bis Ende -> best effort: liquidate am buy_dt zum Buy-Preis
            sell_dt[sym] = buy_dt
            sell_px[sym] = float(valid[sym])
            reason[sym] = "no_price_upto_end_for_symbol"
        else:
            sell_dt[sym] = pd.Timestamp(dt_last)
            sell_px[sym] = float(px_all.at[dt_last, sym])
            reason[sym] = "end_date_liquidation" if pd.Timestamp(dt_last) == pd.Timestamp(end_date) else "delisted_or_no_price_before_end"

    # Positionswerte bis Sell-Date
    px_sim = px.loc[:, w.index]
    pos_val = px_sim.mul(shares, axis=1)

    for sym in w.index:
        pos_val.loc[pos_val.index > sell_dt[sym], sym] = 0.0

    # Cash ab Sell-Date
    cash = pd.Series(0.0, index=pos_val.index)
    for sym in w.index:
        proceeds = float(shares[sym] * sell_px[sym])
        cash.loc[cash.index >= sell_dt[sym]] += proceeds

    nav = pos_val.sum(axis=1) + cash
    df_nav = pd.DataFrame({"date": pos_val.index, "nav": nav.values, "cash": cash.values})

    # Trade Log
    rows: List[Dict[str, Any]] = []
    for sym in w.index:
        rows.append(
            {
                "symbol": sym,
                "side": "BUY",
                "trade_date": buy_dt,
                "price": float(valid[sym]),
                "shares": float(shares[sym]),
                "notional": float(shares[sym] * float(valid[sym])),
                "reason": "buy_like_dynamic_same_weights",
            }
        )

    for sym in w.index:
        rows.append(
            {
                "symbol": sym,
                "side": "SELL",
                "trade_date": sell_dt[sym],
                "price": float(sell_px[sym]),
                "shares": float(shares[sym]),
                "notional": float(shares[sym] * float(sell_px[sym])),
                "reason": reason[sym],
            }
        )

    for sym in invalid_syms:
        rows.append(
            {
                "symbol": sym,
                "side": "SKIP",
                "trade_date": buy_dt,
                "price": None,
                "shares": 0.0,
                "notional": 0.0,
                "reason": "missing_buy_price_on_buy_dt",
            }
        )

    df_trades = pd.DataFrame(rows).sort_values(["trade_date", "symbol", "side"]).reset_index(drop=True)
    return df_nav, df_trades


# ============================================================
# Dynamic (Quarterly switch) helpers + simulation
# ============================================================
def sort_portfolios_quarterly(portfolios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tmp = []
    for p in portfolios:
        k = parse_portfolio_name(p.get("name", ""))
        if k:
            tmp.append((k, p))
    tmp.sort(key=lambda x: x[0])
    return [p for _, p in tmp]


def price_on_or_before(
    es_client,
    prices_index: str,
    symbol: str,
    ld: pd.Timestamp,
) -> Optional[float]:
    ld = pd.to_datetime(ld, utc=True, errors="coerce")
    if pd.isna(ld):
        return None
    ld_str = ld.strftime("%Y-%m-%d")

    for sym_field in ["symbol"]:
        body = {
            "size": 1,
            "_source": ["symbol", "date", "adjClose"],
            "query": {"bool": {"filter": [{"term": {sym_field: symbol}}, {"range": {"date": {"lte": ld_str}}}]}},
            "sort": [{"date": "desc"}],
        }
        try:
            resp = es_client.search(index=prices_index, body=body)
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
    _es_client,  # Streamlit hasht den ES-Client NICHT wegen leading underscore
    portfolio_minimal: List[Dict[str, str]],
    initial_capital: float,
    tax_rate: float,
    year_from: int,
    year_to: int,
    prices_index: str,
    stocks_index: str,
) -> pd.DataFrame:
    """
    Quartals-Umschichtung:
    - Portfolios nach Name YYYY-Qx sortieren
    - ld = max(date) je Symbol aus stocks (calendarYear+period) → global max
    - Verkauf alte Holdings am ld (Bewertung) + Steuer auf Gewinne
    - Reinvest nach neuen Gewichten am selben ld
    """
    ports_full = [{"id": p["id"], "name": p.get("name", "")} for p in portfolio_minimal if p.get("id")]
    ports = sort_portfolios_quarterly(ports_full)
    ports = [p for p in ports if parse_portfolio_name(p.get("name", "")) is not None]
    ports = [p for p in ports if year_from <= parse_portfolio_name(p["name"])[0] <= year_to]

    if len(ports) < 2:
        return pd.DataFrame()

    holdings: Dict[str, Dict[str, float]] = {}
    nav_prev = float(initial_capital)

    # --- Initial-Kauf (erstes Quartal) ---
    first_p = ports[0]
    pq0 = parse_portfolio_name(first_p.get("name", ""))
    if pq0 is None:
        return pd.DataFrame()
    y0, q0 = pq0

    from src.funktionen import load_portfolio  # lokal import, um Zirkularität zu vermeiden

    first_doc = load_portfolio(_es_client, first_p.get("id"))
    if not first_doc:
        return pd.DataFrame()

    w0 = portfolio_doc_to_amount_weights(first_doc)
    syms0 = sorted(w0.keys())
    if not syms0:
        return pd.DataFrame()

    ld0 = ld_from_stocks(_es_client, stocks_index, syms0, y0, q0)
    if ld0 is None or pd.isna(ld0):
        return pd.DataFrame()

    holdings = {}
    missing_new_px: List[str] = []
    bought_new: List[str] = []

    for sym, w in w0.items():
        px = price_on_or_before(_es_client, prices_index, sym, ld0)
        if px is None or px <= 0:
            missing_new_px.append(sym)
            continue
        alloc = float(initial_capital) * float(w)
        shares = alloc / px
        holdings[sym] = {"shares": shares, "buy_price": px}
        bought_new.append(sym)

    results = [
        {
            "quarter": first_p.get("name", ""),
            "ld": pd.to_datetime(ld0).tz_convert(None),
            "sell_date": pd.to_datetime(ld0).tz_convert(None),
            "buy_date": pd.to_datetime(ld0).tz_convert(None),
            "start_value": float(initial_capital),
            "gross_value": float(initial_capital),
            "tax": 0.0,
            "end_value": float(initial_capital),
            "return_q": 0.0,
            "n_old": 0,
            "n_new": len(holdings),
            "new_target_syms": ", ".join(syms0),
            "new_bought_syms": ", ".join(sorted(bought_new)),
            "new_missing_px_syms": ", ".join(sorted(missing_new_px)),
            "old_missing_px_syms": "",
        }
    ]

    # --- Folgequartale ---
    for idx in range(1, len(ports)):
        neu_p = ports[idx]
        neu_name = neu_p.get("name", "")
        pq = parse_portfolio_name(neu_name)
        if pq is None:
            continue
        year, quarter = pq

        neu_doc = load_portfolio(_es_client, neu_p.get("id"))
        if not neu_doc:
            continue

        weights = portfolio_doc_to_amount_weights(neu_doc)
        new_syms = sorted(weights.keys())
        if not new_syms:
            continue

        ld = ld_from_stocks(_es_client, stocks_index, new_syms, year, quarter)
        if ld is None or pd.isna(ld):
            continue

        gross_value = 0.0
        tax = 0.0

        # Verkauf/Valuation alte Holdings
        missing_old_px: List[str] = []
        if holdings:
            for sym, pos in holdings.items():
                px = price_on_or_before(_es_client, prices_index, sym, ld)
                if px is None:
                    missing_old_px.append(sym)
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

        # Neu kaufen
        new_holdings: Dict[str, Dict[str, float]] = {}
        missing_new_px = []
        bought_new = []

        for sym, w in weights.items():
            px = price_on_or_before(_es_client, prices_index, sym, ld)
            if px is None or px <= 0:
                missing_new_px.append(sym)
                continue
            alloc = end_value * float(w)
            shares = alloc / px
            new_holdings[sym] = {"shares": shares, "buy_price": px}
            bought_new.append(sym)

        results.append(
            {
                "quarter": neu_name,
                "ld": pd.to_datetime(ld).tz_convert(None),
                "sell_date": pd.to_datetime(ld).tz_convert(None),
                "buy_date": pd.to_datetime(ld).tz_convert(None),
                "start_value": nav_prev,
                "gross_value": gross_value,
                "tax": tax,
                "end_value": end_value,
                "return_q": ret_q,
                "n_old": len(holdings),
                "n_new": len(new_holdings),
                "new_target_syms": ", ".join(new_syms),
                "new_bought_syms": ", ".join(sorted(bought_new)),
                "new_missing_px_syms": ", ".join(sorted(missing_new_px)),
                "old_missing_px_syms": ", ".join(sorted(missing_old_px)),
            }
        )

        holdings = new_holdings
        nav_prev = end_value

    df = pd.DataFrame(results)
    if df.empty:
        return df

    for c in ["start_value", "gross_value", "tax", "end_value", "return_q"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ld"] = pd.to_datetime(df["ld"], errors="coerce")
    df["sell_date"] = pd.to_datetime(df["sell_date"], errors="coerce")
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")

    df = df.dropna(subset=["ld", "end_value"]).sort_values("ld")
    return df


# ============================================================
# Quarter helpers (ONE canonical implementation)
# ============================================================
def quarter_to_index(qname: str) -> Optional[int]:
    pq = parse_portfolio_name(qname)
    if pq is None:
        return None
    y, q = pq
    return y * 4 + (q - 1)


def index_to_quarter(qi: int) -> str:
    y = qi // 4
    q = (qi % 4) + 1
    return f"{y}-Q{q}"


def quarter_end_ts(qname: str) -> Optional[pd.Timestamp]:
    pq = parse_portfolio_name(qname)
    if pq is None:
        return None
    y, q = pq
    if q == 1:
        return pd.Timestamp(year=y, month=3, day=31)
    if q == 2:
        return pd.Timestamp(year=y, month=6, day=30)
    if q == 3:
        return pd.Timestamp(year=y, month=9, day=30)
    return pd.Timestamp(year=y, month=12, day=31)


def list_quarters_between(q_start: str, q_end: str) -> List[str]:
    i0 = quarter_to_index(q_start)
    i1 = quarter_to_index(q_end)
    if i0 is None or i1 is None or i1 < i0:
        return []
    return [index_to_quarter(i) for i in range(i0, i1 + 1)]


def nav_to_eoq(df_nav: pd.DataFrame) -> pd.DataFrame:
    """
    df_nav: columns ['date','nav'] (daily)
    returns: quarter, ld (EOQ date), end_value (EOQ nav), q_index
    EOQ = letzter Handelstag im Kalenderquartal.
    """
    if df_nav is None or df_nav.empty:
        return pd.DataFrame()

    x = df_nav.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["nav"] = pd.to_numeric(x["nav"], errors="coerce")
    x = x.dropna(subset=["date", "nav"]).sort_values("date")

    x["year"] = x["date"].dt.year
    x["q"] = x["date"].dt.quarter
    x["quarter"] = x["year"].astype(str) + "-Q" + x["q"].astype(str)

    eoq = x.groupby(["year", "q"], as_index=False).tail(1)[["quarter", "date", "nav"]]
    eoq = eoq.rename(columns={"date": "ld", "nav": "end_value"}).sort_values("ld").reset_index(drop=True)
    eoq["q_index"] = eoq["quarter"].map(quarter_to_index)
    return eoq


def geom_avg_quarter_return(v0: float, v1: float, n_quarters: int) -> float:
    """(v1/v0)^(1/n) - 1"""
    if v0 <= 0 or v1 <= 0 or n_quarters <= 0:
        return float("nan")
    return (v1 / v0) ** (1.0 / float(n_quarters)) - 1.0


# ============================================================
# (Optional) Triangle builder for Saved Portfolios (generic)
# ============================================================
def build_saved_portfolios_triangle(
    es_client,
    list_portfolios_fn,
    load_portfolio_fn,
    prices_matrix_loader_fn,
    stocks_index: str,
    end_date_cutoff: pd.Timestamp,
    q_start: str,
    q_end: str,
    base_capital: float = 1000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dreieck über *alle* Saved Portfolios:
    - jede Zeile = Portfolio (Name = Kaufquartal YYYY-Qx)
    - jede Spalte = Verkaufsquartal
    - Zelle = Ø Quartalsrendite (geom.) aus NAV(buy) -> NAV(sell)
    """
    sell_quarters = list_quarters_between(q_start, q_end)
    if not sell_quarters:
        return pd.DataFrame(), pd.DataFrame([{"name": "", "id": "", "status": "bad_quarter_range"}])

    end_dt = quarter_end_ts(q_end)
    if end_dt is None:
        return pd.DataFrame(), pd.DataFrame([{"name": "", "id": "", "status": "bad_q_end"}])

    end_dt = min(pd.Timestamp(end_dt), pd.Timestamp(end_date_cutoff))

    ports = list_portfolios_fn(es_client) or []
    ports_q: List[Dict[str, Any]] = []
    for p in ports:
        pid = p.get("id")
        name = p.get("name", "")
        if pid and parse_portfolio_name(name) is not None:
            ports_q.append({"id": pid, "name": name})

    status_rows: List[Dict[str, Any]] = []
    nav_cache: Dict[str, pd.DataFrame] = {}

    # NAV je Portfolio einmal rechnen
    for p in ports_q:
        pid = p["id"]
        doc = load_portfolio_fn(es_client, pid)
        if not doc:
            status_rows.append({"name": p["name"], "id": pid, "status": "missing_doc"})
            continue

        weights = portfolio_doc_to_amount_weights(doc)
        if not weights:
            status_rows.append({"name": p["name"], "id": pid, "status": "no_weights"})
            continue

        buy_dt = get_buy_date_like_dynamic_for_portfolio(es_client, stocks_index, doc)
        if buy_dt is None:
            status_rows.append({"name": p["name"], "id": pid, "status": "no_buy_dt"})
            continue

        syms = sorted(weights.keys())
        prices_mat = prices_matrix_loader_fn(syms)
        if prices_mat is None or prices_mat.empty:
            status_rows.append({"name": p["name"], "id": pid, "status": "no_prices"})
            continue

        df_nav, _df_trades = build_saved_buyhold_series_with_liquidation(
            prices=prices_mat,
            weights=weights,
            buy_date=buy_dt,
            initial_capital=float(base_capital),
            end_date=end_dt,
        )
        if df_nav is None or df_nav.empty:
            status_rows.append({"name": p["name"], "id": pid, "status": "nav_empty"})
            continue

        df_nav = df_nav.copy()
        df_nav["date"] = pd.to_datetime(df_nav["date"], errors="coerce")
        df_nav["nav"] = pd.to_numeric(df_nav["nav"], errors="coerce")
        df_nav = df_nav.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)

        status_rows.append({"name": p["name"], "id": pid, "status": "ok"})
        nav_cache[pid] = df_nav

    # Dreiecks-Zellen bauen
    rows: List[Dict[str, Any]] = []
    for p in ports_q:
        pid = p["id"]
        buy_q = p["name"]
        df_nav = nav_cache.get(pid)
        if df_nav is None or df_nav.empty:
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
            v1 = float(x["nav"].iloc[-1])

            val = geom_avg_quarter_return(v0, v1, n_q)
            if pd.isna(val):
                continue

            rows.append({"buy_q": buy_q, "sell_q": sell_q, "n_q": n_q, "value": float(val)})

    tri_df = pd.DataFrame(rows)
    status_df = pd.DataFrame(status_rows)
    return tri_df, status_df
