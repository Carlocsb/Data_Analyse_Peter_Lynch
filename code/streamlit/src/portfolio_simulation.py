# src/portfolio_simulation.py
from __future__ import annotations
import streamlit as st

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import re

# =========================
# Naming / Quarter parsing
# =========================
_QRE = re.compile(r"^(?P<y>\d{4})-Q(?P<q>[1-4])$")


def parse_portfolio_name(name: str) -> Optional[tuple[int, int]]:
    if not name:
        return None
    m = _QRE.match(str(name).strip())
    if not m:
        return None
    return int(m.group("y")), int(m.group("q"))


# =========================
# Portfolio weights
# =========================
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


# =========================
# Buy date = like dynamic (ld from stocks)
# =========================
def ld_from_stocks(
    es_client,
    stocks_index: str,
    symbols: List[str],
    year: int,
    quarter: int,
) -> Optional[pd.Timestamp]:
    """
    Pro Symbol max(date) in stocks (calendarYear, period), dann global max.
    Das ist exakt die Logik aus deinem Dynamik-Code.
    """
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
                resp = es_client.search(index=stocks_index, body=body)
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

    return pd.to_datetime(ld).tz_convert(None)


# =========================
# Buy&Hold with liquidation per symbol
# =========================
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
    prices: pd.DataFrame,                 # index=date, columns=symbols (adjClose)
    weights: Dict[str, float],            # sum ~ 1
    buy_date: pd.Timestamp,               # ld wie dynamisch
    initial_capital: float,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simuliert Buy-&-Hold ab buy_date (gemappt auf nächsten Handelstag im prices-Index),
    Liquidation je Symbol:
    - bevorzugt end_date, sonst letzter Preis <= end_date.
    - nach Liquidation: Position -> Cash (Cash bleibt im NAV).
    Gibt zurück:
      df_nav:  date, nav, cash
      df_trades: BUY/SELL/SKIP Log
    """
    if prices is None or prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    px_all = prices.sort_index().copy()
    idx_all = px_all.index

    # mappe buy_date auf nächsten Handelstag im globalen Index
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
        rows.append({
            "symbol": sym,
            "side": "BUY",
            "trade_date": buy_dt,
            "price": float(valid[sym]),
            "shares": float(shares[sym]),
            "notional": float(shares[sym] * float(valid[sym])),
            "reason": "buy_like_dynamic_same_weights",
        })

    for sym in w.index:
        rows.append({
            "symbol": sym,
            "side": "SELL",
            "trade_date": sell_dt[sym],
            "price": float(sell_px[sym]),
            "shares": float(shares[sym]),
            "notional": float(shares[sym] * float(sell_px[sym])),
            "reason": reason[sym],
        })

    for sym in invalid_syms:
        rows.append({
            "symbol": sym,
            "side": "SKIP",
            "trade_date": buy_dt,
            "price": None,
            "shares": 0.0,
            "notional": 0.0,
            "reason": "missing_buy_price_on_buy_dt",
        })

    df_trades = pd.DataFrame(rows).sort_values(["trade_date", "symbol", "side"]).reset_index(drop=True)
    return df_nav, df_trades
# =========================
# Dynamic (Quarterly switch) helpers + simulation
# =========================

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
    _es_client,  # wichtig: führender underscore => Streamlit hasht den ES-Client NICHT
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

    # load_portfolio ist in src.funktionen => import lokal, um Zirkularität zu vermeiden
    from src.funktionen import load_portfolio

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
    missing_new_px = []
    bought_new = []

    for sym, w in w0.items():
        px = price_on_or_before(_es_client, prices_index, sym, ld0)
        if px is None or px <= 0:
            missing_new_px.append(sym)
            continue
        alloc = float(initial_capital) * float(w)
        shares = alloc / px
        holdings[sym] = {"shares": shares, "buy_price": px}
        bought_new.append(sym)

    results = [{
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
    }]

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
        missing_old_px = []
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

        results.append({
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
        })

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
