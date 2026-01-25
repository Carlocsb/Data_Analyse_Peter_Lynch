import os
import pandas as pd
from elasticsearch import Elasticsearch, NotFoundError
import streamlit as st
from typing import Optional, Dict, Any
import plotly.express as px
from datetime import datetime, timezone
import re
from .lynch_criteria import CATEGORIES

# ==========================================================
# 1️⃣ ELASTICSEARCH CONNECTION AND DATA RETRIEVAL
# ==========================================================

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX  = os.getenv("ELASTICSEARCH_INDEX", "stocks")

# --------- NEW: History helpers for field alias lists ---------
def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _merge_asof_two(left: pd.DataFrame, right: pd.DataFrame):
    """Expects columns ['Date','Value']; performs asof-join and returns merged frame."""
    if left.empty or right.empty:
        return pd.DataFrame()
    l = left.sort_values("Date")
    r = right.sort_values("Date")
    m = pd.merge_asof(l, r, on="Date", direction="nearest")
    m = m.dropna()
    return m



def _safe_sheet_name(name: str) -> str:
    # Excel: max 31, keine : \ / ? * [ ]
    name = (name or "Portfolio").strip()
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    return name[:31] if len(name) > 31 else name


def export_all_portfolios_to_excel_scroll(
    es,
    index_name: str,
    filepath: str = "exports/portfolios_export.xlsx",
    scroll: str = "2m",
    batch_size: int = 500,
) -> str:
    """
    Exportiert ALLE Portfolios aus Elasticsearch (vollständig) in eine Excel:
    - Sheet 'All' mit allem
    - pro Portfolio ein Sheet

    Erwartet Dokument-Struktur:
      doc = { "name":..., "market_condition":..., "selected_industry":..., "items":[{"category","symbol","amount"}, ...] }
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 1) initial search (scroll)
    resp = es.search(
        index=index_name,
        body={"query": {"match_all": {}}, "sort": ["_doc"]},
        size=batch_size,
        scroll=scroll,
    )

    scroll_id = resp.get("_scroll_id")
    hits = resp.get("hits", {}).get("hits", [])

    all_rows = []
    used_sheet_names = set()

    def doc_to_df(doc_id: str, src: dict) -> pd.DataFrame:
        pname = src.get("name") or f"portfolio_{doc_id}"
        market = src.get("market_condition", "")
        industry = src.get("selected_industry", "")

        items = src.get("items") or []
        # falls items mal als dict kommt -> in liste umformen
        if isinstance(items, dict):
            # best effort: {category:{symbol:amount}} oder ähnlich
            tmp = []
            for cat, symvals in items.items():
                if isinstance(symvals, dict):
                    for sym, amt in symvals.items():
                        tmp.append({"category": cat, "symbol": sym, "amount": amt})
            items = tmp

        df = pd.DataFrame(items)
        if df.empty:
            df = pd.DataFrame(columns=["category", "symbol", "amount"])

        for col in ["category", "symbol", "amount"]:
            if col not in df.columns:
                df[col] = None

        df["portfolio_id"] = doc_id
        df["portfolio_name"] = pname
        df["market_condition"] = market
        df["selected_industry"] = industry

        df = df[["portfolio_name", "portfolio_id", "market_condition", "selected_industry", "category", "symbol", "amount"]]
        df = df.sort_values(by=["category", "symbol"], kind="stable")
        return df

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # 2) iterate scroll pages
        while hits:
            for h in hits:
                doc_id = h.get("_id")
                src = h.get("_source", {}) or {}

                df = doc_to_df(doc_id, src)
                all_rows.append(df)

                # Sheetname: NAME + short ID => garantiert eindeutig, keine Überschreibung
                base = f"{src.get('name','Portfolio')}_{doc_id[:6]}"
                sheet = _safe_sheet_name(base)
                if sheet in used_sheet_names:
                    # notfalls eindeutiger machen
                    sheet = _safe_sheet_name(f"{src.get('name','Portfolio')}_{doc_id[:10]}")
                used_sheet_names.add(sheet)

                df.to_excel(writer, sheet_name=sheet, index=False)

            # next scroll
            resp = es.scroll(scroll_id=scroll_id, scroll=scroll)
            scroll_id = resp.get("_scroll_id")
            hits = resp.get("hits", {}).get("hits", [])

        # 3) "All" sheet
        if all_rows:
            df_all = pd.concat(all_rows, ignore_index=True)
            df_all = df_all.sort_values(by=["portfolio_name", "category", "symbol"], kind="stable")
        else:
            df_all = pd.DataFrame(columns=["portfolio_name","portfolio_id","market_condition","selected_industry","category","symbol","amount"])

        df_all.to_excel(writer, sheet_name="All", index=False)

    # scroll cleanup (optional)
    try:
        if scroll_id:
            es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    return filepath

def get_es_connection():
    """Creates a connection to Elasticsearch."""
    es = Elasticsearch(ES_URL, request_timeout=30)
    if not es.ping():
        print("❌ Failed to connect to Elasticsearch.")
    return es


# -------- Central source selection & dedupe control --------
SOURCE_MODES = [
    "Only yfinance",
    "Only Alpha Vantage",
    "Only FMP",
    "Both – prefer yfinance",
    "Both – newest ingest wins",
]


def render_source_selector(label: str = "📡 Data source") -> str:
    """Sidebar toggle (global via Session State)."""
    if "src_mode" not in st.session_state:
        st.session_state["src_mode"] = "Only FMP"  # default is FMP now
    return st.sidebar.radio(label, SOURCE_MODES, key="src_mode")


def _term(field: str, value: str) -> Dict[str, Any]:
    """Helper: term query with .keyword fallback."""
    return {
        "bool": {
            "should": [
                {"term": {f"{field}.keyword": value}},
                {"term": {field: value}},
            ],
            "minimum_should_match": 1,
        }
    }


def _es_query_for_mode(mode: Optional[str]) -> Optional[Dict[str, Any]]:
    """For single-source modes, filter already in the ES query."""
    if mode == "Only yfinance":
        return _term("source", "yfinance")
    if mode == "Only Alpha Vantage":
        return _term("source", "alphavantage")
    if mode == "Only FMP":
        return _term("source", "fmp")
    return None


def _filter_dedupe_by_mode(df: pd.DataFrame, mode: Optional[str]) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return df

    out = df.copy()
    if "ingested_at" in out.columns:
        out["ingested_at"] = pd.to_datetime(out["ingested_at"], utc=True, errors="coerce")

    if mode == "Only yfinance":
        return out[out["source"] == "yfinance"]
    if mode == "Only Alpha Vantage":
        return out[out["source"] == "alphavantage"]
    if mode == "Only FMP":
        return out[out["source"] == "fmp"]

    # Combined modes: reduce to 1 row per (symbol, date)
    if mode == "Both – prefer yfinance":
        # Order: yfinance → fmp → alphavantage → other
        pref = {"yfinance": 0, "fmp": 1, "alphavantage": 2}
        out["__src_rank"] = out["source"].map(pref).fillna(9)
        out = (
            out.sort_values(["symbol", "date", "__src_rank", "ingested_at"])
               .drop_duplicates(subset=["symbol", "date"], keep="first")
               .drop(columns=["__src_rank"])
        )
        return out

    if mode == "Both – newest ingest wins":
        if "ingested_at" in out.columns:
            return (
                out.sort_values(["symbol", "date", "ingested_at"])
                   .drop_duplicates(subset=["symbol", "date"], keep="last")
            )
        return out.drop_duplicates(subset=["symbol", "date"], keep="last")

    return out


# ==========================================================
# 1b️⃣ Enrichment / Derived metrics
# ==========================================================

def _safe_div(a, b):
    try:
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _first_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _get_any(d: dict, *keys):
    """First non-None value from d for a list of possible keys."""
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def load_historical_metrics(es, symbol: str, metric, source_mode: Optional[str] = None):
    """
    Loads history for a metric or a list of possible field names (aliases).
    Returns DF: columns ['Date','Value'] (sorted ascending).
    """
    symbol = (symbol or "").strip().upper()
    fields = _ensure_list(metric)

    must = [_term("symbol", symbol)]
    q_src = _es_query_for_mode(source_mode)
    if q_src:
        must.append(q_src)

    # fetch all candidate fields at once and pick first with actual data
    _source = list(dict.fromkeys(["symbol", "date", "source", "ingested_at", *fields]))

    query = {
        "size": 10000,
        "query": {"bool": {"must": must}},
        "sort": [{"date": {"order": "asc"}}, {"ingested_at": {"order": "asc"}}],
        "_source": _source,
    }
    resp = es.search(index=INDEX, body=query)
    hits = [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
    raw = pd.DataFrame(hits)
    if raw.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = _filter_dedupe_by_mode(raw, source_mode)
    raw = raw.sort_values("date")

    # pick first field with real values
    for f in fields:
        if f in raw.columns and raw[f].notna().any():
            out = raw[["date", f]].dropna()
            if not out.empty:
                out = out.rename(columns={"date": "Date", f: "Value"})
                return out
    return pd.DataFrame(columns=["Date", "Value"])


def _compute_yoy_from_history(es, symbol: str, fields, source_mode: Optional[str] = None, periods_back: int = 4):
    """
    YoY (current quarter vs prior-year quarter) for a field OR a list of fields.
    For a field list, uses the first field that has data.
    """
    df_hist = load_historical_metrics(es, symbol, fields, source_mode)
    if df_hist.empty or len(df_hist) <= periods_back:
        return None
    latest = _first_float(df_hist["Value"].iloc[-1])
    prev = _first_float(df_hist["Value"].iloc[-1 - periods_back])
    if latest is None or prev in (None, 0):
        return None
    return (latest - prev) / abs(prev)


def enrich_document_fields(doc: dict, es=None, source_mode: Optional[str] = None, fill_growth_from_history: bool = True) -> dict:
    """
    Fills ONLY missing fields (None).
    - maps FMP aliases -> standard fields
    - computes useful derivations (per-share, margins, ratios)
    - optional: YoY growth from ES history
    """
    d = dict(doc)  # copy

    # --- 0) Aliases (FMP -> standard names), ONLY if standard field missing ---
    # Multiples / Ratios
    if d.get("peRatio") is None:
        d["peRatio"] = _get_any(d, "priceEarningsRatio", "trailingPE", "peRatio")
    if d.get("priceToBook") is None:
        d["priceToBook"] = _get_any(d, "priceToBookRatio")
    if d.get("pegRatio") is None:
        d["pegRatio"] = _get_any(d, "priceEarningsToGrowthRatio")

    # Dividends
    if d.get("dividendYield") is None:
        d["dividendYield"] = _get_any(d, "dividendYield")
    if d.get("payoutRatio") is None:
        d["payoutRatio"] = _get_any(d, "payoutRatio")

    # Profile/Meta
    if d.get("beta") is None:
        d["beta"] = _get_any(d, "beta")
    if d.get("marketCap") is None:
        d["marketCap"] = _get_any(d, "marketCap", "mktCap")
    if d.get("industry") is None:
        d["industry"] = _get_any(d, "industry")
    if d.get("sector") is None:
        d["sector"] = _get_any(d, "sector")

    # Income Statement
    if d.get("revenue") is None:
        d["revenue"] = _get_any(d, "revenue")
    if d.get("netIncome") is None:
        d["netIncome"] = _get_any(d, "netIncome")
    if d.get("eps") is None:
        d["eps"] = _get_any(d, "eps", "epsdiluted", "epsDiluted")

    # Balance Sheet
    if d.get("totalAssets") is None:
        d["totalAssets"] = _get_any(d, "totalAssets")
    if d.get("totalStockholderEquity") is None:
        d["totalStockholderEquity"] = _get_any(
            d, "totalStockholderEquity", "totalStockholdersEquity", "shareholdersEquity"
        )
    if d.get("totalDebt") is None:
        d["totalDebt"] = _get_any(d, "totalDebt")
    if d.get("totalCash") is None:
        d["totalCash"] = _get_any(d, "cashAndShortTermInvestments", "cashAndCashEquivalents")
    if d.get("sharesOutstanding") is None:
        d["sharesOutstanding"] = _get_any(
            d, "sharesOutstanding", "weightedAverageShsOut", "weightedAverageShsOutDil"
        )
    if d.get("totalCurrentAssets") is None:
        d["totalCurrentAssets"] = _get_any(d, "totalCurrentAssets")
    if d.get("totalCurrentLiabilities") is None:
        d["totalCurrentLiabilities"] = _get_any(d, "totalCurrentLiabilities")
    if d.get("inventory") is None:
        d["inventory"] = _get_any(d, "inventory")

    # Ratios/KeyMetrics per share
    if d.get("cashPerShare") is None:
        d["cashPerShare"] = _get_any(d, "cashPerShare")
    if d.get("bookValuePerShare") is None:
        d["bookValuePerShare"] = _get_any(
            d, "bookValuePerShare", "shareholdersEquityPerShare", "tangibleBookValuePerShare"
        )
    if d.get("freeCashFlowPerShare") is None:
        d["freeCashFlowPerShare"] = _get_any(d, "freeCashFlowPerShare")

    # Cash Flow Statement → OCF / CapEx / FCF
    if d.get("operatingCashflow") is None and d.get("operatingCashFlow") is not None:
        d["operatingCashflow"] = d.get("operatingCashFlow")
    if d.get("capitalExpenditures") is None and d.get("capitalExpenditure") is not None:
        d["capitalExpenditures"] = d.get("capitalExpenditure")
    if d.get("freeCashFlow") is None:
        d["freeCashFlow"] = _get_any(d, "freeCashFlow", "freeCashflow")

    # --- 1) Direct DERIVATIONS (only if target missing) ---
    if d.get("profitMargin") is None and d.get("netIncome") is not None and d.get("revenue"):
        d["profitMargin"] = _safe_div(d["netIncome"], d["revenue"])

    if d.get("currentRatio") is None and d.get("totalCurrentAssets") is not None and d.get("totalCurrentLiabilities"):
        d["currentRatio"] = _safe_div(d["totalCurrentAssets"], d["totalCurrentLiabilities"])

    if d.get("quickRatio") is None and all(k in d for k in ("totalCurrentAssets", "inventory", "totalCurrentLiabilities")):
        ca = _first_float(d.get("totalCurrentAssets"))
        inv = _first_float(d.get("inventory"))
        cl  = _first_float(d.get("totalCurrentLiabilities"))
        if ca is not None and inv is not None and cl not in (None, 0):
            d["quickRatio"] = (ca - inv) / cl

    if d.get("cashToDebt") is None and d.get("totalCash") is not None and d.get("totalDebt"):
        d["cashToDebt"] = _safe_div(d["totalCash"], d["totalDebt"])

    if d.get("equityRatio") is None and d.get("totalStockholderEquity") is not None and d.get("totalAssets"):
        d["equityRatio"] = _safe_div(d["totalStockholderEquity"], d["totalAssets"])

    if d.get("debtToAssets") is None and d.get("totalDebt") is not None and d.get("totalAssets"):
        d["debtToAssets"] = _safe_div(d["totalDebt"], d["totalAssets"])

    if d.get("debtToEquity") is None and d.get("totalDebt") is not None and d.get("totalStockholderEquity"):
        d["debtToEquity"] = _safe_div(d["totalDebt"], d["totalStockholderEquity"])

    if d.get("bookValuePerShare") is None and d.get("totalStockholderEquity") is not None and d.get("sharesOutstanding"):
        d["bookValuePerShare"] = _safe_div(d["totalStockholderEquity"], d["sharesOutstanding"])

    if d.get("cashPerShare") is None and d.get("totalCash") is not None and d.get("sharesOutstanding"):
        d["cashPerShare"] = _safe_div(d["totalCash"], d["sharesOutstanding"])

    if d.get("freeCashFlowPerShare") is None and d.get("freeCashFlow") is not None and d.get("sharesOutstanding"):
        d["freeCashFlowPerShare"] = _safe_div(d["freeCashFlow"], d["sharesOutstanding"])

    if d.get("fcfMargin") is None and d.get("freeCashFlow") is not None and d.get("revenue"):
        d["fcfMargin"] = _safe_div(d["freeCashFlow"], d["revenue"])

    # Price/Book from MarketCap/Equity (fallback)
    if d.get("priceToBook") is None and d.get("marketCap") is not None and d.get("totalStockholderEquity"):
        d["priceToBook"] = _safe_div(d["marketCap"], d["totalStockholderEquity"])

    # EPS (fallback) from NetIncome / Shares
    if d.get("eps") is None and d.get("netIncome") is not None and d.get("sharesOutstanding"):
        d["eps"] = _safe_div(d["netIncome"], d["sharesOutstanding"])

    # PEG (PE / earningsGrowth in percent- or decimal-form)
    if d.get("pegRatio") is None and d.get("peRatio") is not None and d.get("earningsGrowth") not in (None, 0):
        try:
            eg = float(d.get("earningsGrowth"))
            denom = eg * 100.0 if abs(eg) < 1 else eg
            if denom:
                d["pegRatio"] = float(d.get("peRatio")) / denom
        except Exception:
            pass

    # --- 2) YoY growth from history (only if missing & desired) ---
    symbol = d.get("symbol")
    if fill_growth_from_history and es is not None and symbol:
        if d.get("revenueGrowth") is None:
            d["revenueGrowth"] = _compute_yoy_from_history(es, symbol, "revenue", source_mode)
        if d.get("epsGrowth") is None:
            d["epsGrowth"] = _compute_yoy_from_history(es, symbol, "eps", source_mode)
        if d.get("earningsGrowth") is None:
            d["earningsGrowth"] = d.get("epsGrowth")

    return d


# ==========================================================
# 1c️⃣ Search (with enrichment)
# ==========================================================

def search_stock_in_es(es, symbol: str, source_mode: Optional[str] = None):
    symbol = (symbol or "").strip().upper()

    must = [
        {
            "bool": {
                "should": [
                    {"term": {"symbol.keyword": symbol}},
                    {"term": {"symbol": symbol}},
                    {"match": {"symbol": symbol}},
                ],
                "minimum_should_match": 1,
            }
        }
    ]
    q_src = _es_query_for_mode(source_mode)
    if q_src:
        must.append(q_src)

    query = {
        "size": 1000,
        "query": {"bool": {"must": must}},
        "sort": [{"date": {"order": "desc"}}, {"ingested_at": {"order": "desc"}}],
    }
    resp = es.search(index=INDEX, body=query)
    hits = [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
    if not hits:
        return None

    df = pd.DataFrame(hits)
    df = _filter_dedupe_by_mode(df, source_mode)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    doc = df.iloc[-1].to_dict()
    doc = enrich_document_fields(doc, es=es, source_mode=source_mode, fill_growth_from_history=True)
    return doc


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_metric_history(df: pd.DataFrame, symbol: str, title: str, unit: str = ""):
    """Creates a Plotly chart for a metric."""
    if df.empty:
        return None
    fig = px.line(df, x="Date", y="Value", title=f"{title} history for {symbol}", markers=True)
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    if unit:
        fig.update_yaxes(title_text=unit)
    return fig


# ==========================================================
#  PETER LYNCH CATEGORIZATION
# ==========================================================

SPECIAL_FIELDS_STRICT = {
    "peRatio",
    "earningsGrowth",
    "epsGrowth",
    "revenueGrowth",
    "dividendYield",
    "freeCashFlowPerShare",
    "freeCashFlow",
    "profitMargin",
}


def score_row(row_or_dict, criteria):
    """
    Scores a stock against the Lynch criteria.

    Rules:
    - max score always equals len(criteria) (no dynamic adjustment)
    - peRatio, earningsGrowth, epsGrowth:
        if value <= 0 → automatically 'not satisfied', even if rule(...) would be True
    """

    getv = (row_or_dict.get if isinstance(row_or_dict, dict) else row_or_dict.__getitem__)
    score = 0

    for item in criteria:
        try:
            # two possible criteria formats
            if len(item) == 2:
                field, rule = item
            elif len(item) >= 3:
                field, _, rule, *rest = item
            else:
                continue

            # fetch value
            val = getv(field)
            if not isinstance(val, (int, float)) or pd.isna(val):
                continue

            # special case: PE & growth → <=0 always false
            if field in SPECIAL_FIELDS_STRICT and val <= 0:
                continue  # no score++

            # regular rule evaluation
            try:
                if rule(val):
                    score += 1
            except Exception:
                continue

        except Exception:
            continue

    return score


def calculate_peter_lynch_category(data: dict, equality_threshold: float = 0.02):
    results = {}
    for cat, rules in CATEGORIES.items():
        score = score_row(data, rules)
        max_rules = len(rules) if rules else 1
        results[cat] = score / max_rules

    # best score
    best_score = max(results.values())
    hit_rate = round(best_score * 100, 1)

    # categories that are nearly as good
    top_categories = [
        cat for cat, s in results.items()
        if best_score - s <= equality_threshold
    ]

    # display text: one or multiple categories
    if len(top_categories) == 1:
        categories_text = top_categories[0]
    elif len(top_categories) == 2:
        categories_text = f"{top_categories[0]} or {top_categories[1]}"
    else:
        categories_text = ", ".join(top_categories[:-1]) + f" or {top_categories[-1]}"
    return categories_text, hit_rate


def explain_category(category: str) -> str:
    texts = {
        "Slow Growers": "💤 Slow-growing companies with stable dividends. Suitable for defensive investors.",
        "Stalwarts": "💪 Established companies with solid growth. Lower risk, still with potential.",
        "Fast Growers": "🚀 Rapid earnings growth. High return potential, but higher risk.",
        "Cyclicals": "🔄 Cyclical companies – strongly dependent on economic cycles.",
        "Turnarounds": "🔁 Companies in recovery – risky, but with significant upside.",
        "Asset Plays": "💎 Companies with hidden assets undervalued by the market.",
    }
    return texts.get(category, "")


# ==========================================================
# HELPERS / UTILITIES
# ==========================================================

def build_metrics_table(data: dict) -> pd.DataFrame:
    details = {
        "Free Cash Flow": data.get("freeCashflow") if "freeCashflow" in data else data.get("freeCashFlow"),
        "Revenue Growth": data.get("revenueGrowth"),
        "Profit Margin": data.get("profitMargin"),
        "Total Debt": data.get("totalDebt"),
        "Quick Ratio": data.get("quickRatio"),
        "Current Ratio": data.get("currentRatio"),
        "Cash / Share": data.get("cashPerShare"),
        "Beta": data.get("beta"),
    }
    df = pd.DataFrame(details.items(), columns=["Metric", "Value"])
    df["Value"] = df["Value"].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else "—")
    return df


def describe_metrics() -> dict:
    return {
        "peRatio": "The price/earnings ratio (P/E) shows how much investors pay for 1 USD of earnings.",
        "priceToBook": "The price-to-book ratio (P/B) compares the share price to the book value.",
        "dividendYield": "Dividend yield shows what percentage dividend is paid per year.",
        "eps": "Earnings per share (EPS) measures profit per share.",
        "bookValuePerShare": "Book value per share reflects the equity value per share.",
        "debtToEquity": "Debt-to-equity ratio; lower values imply lower financial risk.",
    }


# ==========================================================
# DATA LOADING & SCORING – FOR PORTFOLIO AND TOP-10 PAGE
# ==========================================================

def load_data_from_es(es=None, limit: int = 2000, index: str = INDEX, source_mode: Optional[str] = None) -> pd.DataFrame:
    if es is None:
        es = get_es_connection()

    base_query: Dict[str, Any] = {"match_all": {}}
    q_src = _es_query_for_mode(source_mode)
    if q_src:
        base_query = {"bool": {"must": [q_src]}}

    query = {
        "size": limit,
        "sort": [{"date": {"order": "desc"}}, {"ingested_at": {"order": "desc"}}],
        "query": base_query,
    }
    resp = es.search(index=index, body=query)
    hits = [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
    df = pd.DataFrame(hits)
    if df.empty:
        return df

    # latest per symbol
    if "symbol" in df.columns and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["symbol", "date"], ascending=[True, False]).drop_duplicates("symbol", keep="first")

    # Harmonization / derived fields (simple)
    if "marketCap" in df.columns:
        df["marketCapBn"] = df["marketCap"] / 1e9
    if "earningsGrowth" in df.columns and "epsGrowth" not in df.columns:
        df["epsGrowth"] = df["earningsGrowth"]
    if "peRatio" not in df.columns and "trailingPE" in df.columns:
        df["peRatio"] = df["trailingPE"]
    if "freeCashFlowPerShare" not in df.columns and {"freeCashflow", "sharesOutstanding"} <= set(df.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            df["freeCashFlowPerShare"] = df["freeCashflow"] / df["sharesOutstanding"]
    if "fcfMargin" not in df.columns and {"freeCashflow", "revenue"} <= set(df.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            df["fcfMargin"] = df["freeCashflow"] / df["revenue"]
    if "cashToDebt" not in df.columns and {"totalCash", "totalDebt"} <= set(df.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            df["cashToDebt"] = df["totalCash"] / df["totalDebt"]
    if "equityRatio" not in df.columns and {"totalStockholderEquity", "totalAssets"} <= set(df.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            df["equityRatio"] = df["totalStockholderEquity"] / df["totalAssets"]
    if "debtToAssets" not in df.columns and {"totalDebt", "totalAssets"} <= set(df.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            df["debtToAssets"] = df["totalDebt"] / df["totalAssets"]

    # Source/Dedupe
    df = _filter_dedupe_by_mode(df, source_mode)

    # Per-document enrichment (without expensive YoY lookups)
    if not df.empty:
        df = pd.DataFrame([
            enrich_document_fields(r.to_dict(), es=es, source_mode=source_mode, fill_growth_from_history=False)
            for _, r in df.iterrows()
        ])

    return df


def load_industries(es=None, index: str = INDEX, source_mode: Optional[str] = None) -> pd.DataFrame:
    """
    Loads symbol & industry without 'collapse', so it also works
    if 'symbol.keyword' does not exist. We dedupe in Python.
    """
    if es is None:
        es = get_es_connection()

    must = []
    q_src = _es_query_for_mode(source_mode)
    if q_src:
        must.append(q_src)

    query = {
        "size": 10000,
        "_source": ["symbol", "industry"],
        "query": {"bool": {"must": must}} if must else {"match_all": {}},
        "sort": [{"date": {"order": "desc"}}]
    }

    resp = es.search(index=index, body=query)
    hits = [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
    df = pd.DataFrame(hits) if hits else pd.DataFrame(columns=["symbol", "industry"])

    # Dedupe in Python: newest entry per symbol
    if not df.empty and "symbol" in df.columns:
        df = (
            df.dropna(subset=["symbol"])
              .drop_duplicates(subset=["symbol"], keep="first")
        )

    return df[["symbol", "industry"]] if not df.empty else df


# ==========================================================
# 6️⃣ Portfolio functions
# ==========================================================

PORTFOLIO_INDEX = "portfolios"


def ensure_portfolio_index(es):
    if es.indices.exists(index=PORTFOLIO_INDEX):
        return
    es.indices.create(
        index=PORTFOLIO_INDEX,
        body={
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "market_condition": {"type": "keyword"},
                    "industry_filter": {"type": "keyword"},
                    "comment": {"type": "text"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "allocation_target": {"type": "object"},
                    "items": {
                        "type": "nested",
                        "properties": {
                            "category": {"type": "keyword"},
                            "symbol": {"type": "keyword"},
                            "amount": {"type": "double"},
                            "industry": {"type": "keyword"},
                        },
                    },
                    "totals": {
                        "type": "object",
                        "properties": {
                            "by_category": {"type": "object"},
                            "total_amount": {"type": "double"},
                        },
                    },
                }
            }
        },
    )


def build_portfolio_doc(name, market_condition, selected_industry, comment, selected_stocks, amounts, allocation):
    sum_cat = {k: float(sum(amounts.get(k, {}).values())) for k in allocation}
    total_amt = float(sum(sum_cat.values()))
    items = []
    for k, tickers in selected_stocks.items():
        for t in tickers:
            amt = float(amounts.get(k, {}).get(t, 0.0))
            if amt > 0:
                items.append({"category": k, "symbol": t, "amount": amt, "industry": selected_industry})
    now = datetime.now(timezone.utc).isoformat()
    return {
        "name": name,
        "market_condition": market_condition,
        "industry_filter": selected_industry,
        "comment": comment,
        "created_at": now,
        "updated_at": now,
        "allocation_target": allocation,
        "items": items,
        "totals": {"by_category": sum_cat, "total_amount": total_amt},
    }


def save_portfolio(es, doc, portfolio_id=None):
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    if portfolio_id:
        es.update(index=PORTFOLIO_INDEX, id=portfolio_id, body={"doc": doc, "doc_as_upsert": True}, refresh=True)
        return portfolio_id
    res = es.index(index=PORTFOLIO_INDEX, body=doc, refresh=True)
    return res["_id"]


def list_portfolios(es, limit=200):
    resp = es.search(
        index=PORTFOLIO_INDEX,
        body={
            "size": limit,
            "sort": [{"updated_at": {"order": "desc"}}],
            "_source": ["name", "market_condition", "updated_at", "totals.total_amount"],
        },
    )
    return [{"id": h["_id"], **h["_source"]} for h in resp.get("hits", {}).get("hits", [])]


def load_portfolio(es, portfolio_id):
    try:
        return es.get(index=PORTFOLIO_INDEX, id=portfolio_id)["_source"]
    except NotFoundError:
        return None


def delete_portfolio(es, portfolio_id):
    try:
        es.delete(index=PORTFOLIO_INDEX, id=portfolio_id, refresh=True)
        return True
    except NotFoundError:
        return False


def force_fill_metrics(d: dict, es, source_mode: Optional[str] = None, flags: Optional[dict] = None) -> dict:

    if flags is None:
        flags = {
            "revenueGrowth_from_history": True,
            "epsGrowth_from_history": True,
            "fcf_buildable": True,
            "fcf_per_share_buildable": True,
            "debtToAssets_buildable": True,
            "cashToDebt_buildable": True,
            "cashPerShare_buildable": True,
            "bookValuePerShare_buildable": True,
            "sgaTrend_buildable": True,
        }

    out = enrich_document_fields(d, es=es, source_mode=source_mode, fill_growth_from_history=False)

    # 1) YoY via alias lists
    if flags.get("revenueGrowth_from_history") and out.get("revenueGrowth") is None and es is not None:
        out["revenueGrowth"] = _compute_yoy_from_history(
            es, out.get("symbol"),
            fields=["revenue", "totalRevenue", "Revenue", "revenueTTM", "totalRevenueTTM"],
            source_mode=source_mode
        )
    if flags.get("epsGrowth_from_history") and out.get("epsGrowth") is None and es is not None:
        out["epsGrowth"] = _compute_yoy_from_history(
            es, out.get("symbol"),
            fields=["eps", "trailingEps", "reportedEPS", "epsDiluted", "epsdiluted"],
            source_mode=source_mode
        )
    if out.get("earningsGrowth") is None and out.get("epsGrowth") is not None:
        out["earningsGrowth"] = out["epsGrowth"]

    # 2) FCF (OCF - |CapEx|)
    if flags.get("fcf_buildable") and out.get("freeCashFlow") is None:
        ocf   = _get_any(out, "freeCashFlow", "operatingCashflow", "operatingCashFlow",
                         "netCashProvidedByOperatingActivities")
        capex = _get_any(out, "capitalExpenditures", "capitalExpenditure")
        if ocf is not None and capex is not None:
            try:
                out["freeCashFlow"] = float(ocf) - abs(float(capex))
            except Exception:
                pass

    # 3) FCF per share
    if flags.get("fcf_per_share_buildable") and out.get("freeCashFlowPerShare") is None:
        if out.get("freeCashFlow") is not None and out.get("sharesOutstanding"):
            out["freeCashFlowPerShare"] = _safe_div(out["freeCashFlow"], out["sharesOutstanding"])
        # === Backfill totalDebt (e.g., if yfinance missing) ===
    if out.get("totalDebt") is None and es is not None:
        td = load_historical_metrics(
            es, out.get("symbol"),
            ["totalDebt", "shortLongTermDebtTotal", "shortLongTermDebt"],
            source_mode
        )
        if not td.empty:
            out["totalDebt"] = float(td["Value"].iloc[-1])

    # === Backfill totalAssets ===
    if out.get("totalAssets") is None and es is not None:
        ta = load_historical_metrics(
            es, out.get("symbol"),
            ["totalAssets", "TotalAssets"],
            source_mode
        )
        if not ta.empty:
            out["totalAssets"] = float(ta["Value"].iloc[-1])

    # 4) Ratios/derivations
    if flags.get("debtToAssets_buildable") and out.get("debtToAssets") is None:
        if out.get("totalDebt") is not None and out.get("totalAssets") is not None:
            out["debtToAssets"] = _safe_div(out["totalDebt"], out["totalAssets"])

    if flags.get("cashToDebt_buildable") and out.get("cashToDebt") is None:
        if out.get("totalCash") is not None and out.get("totalDebt") is not None:
            out["cashToDebt"] = _safe_div(out["totalCash"], out["totalDebt"])

    if flags.get("cashPerShare_buildable") and out.get("cashPerShare") is None:
        if out.get("totalCash") is not None and out.get("sharesOutstanding"):
            out["cashPerShare"] = _safe_div(out["totalCash"], out["sharesOutstanding"])

    if flags.get("bookValuePerShare_buildable") and out.get("bookValuePerShare") is None:
        if out.get("totalStockholderEquity") is not None and out.get("sharesOutstanding"):
            out["bookValuePerShare"] = _safe_div(out["totalStockholderEquity"], out["sharesOutstanding"])

    # 5) SG&A trend (SG&A/Revenue decreasing?)
    if flags.get("sgaTrend_buildable") and out.get("sgaTrend") is None and es is not None:
        sym = out.get("symbol")
        if sym:
            df_rev = load_historical_metrics(es, sym, ["revenue","totalRevenue","Revenue","revenueTTM"], source_mode)
            df_sga = load_historical_metrics(es, sym, ["sgaExpense","sellingGeneralAndAdministrative","sga"], source_mode)
            m = _merge_asof_two(df_sga, df_rev)
            if not m.empty and len(m) >= 5:
                # m columns: 'Value_x' (SGA), 'Value_y' (Revenue)
                ratio = m["Value_x"] / m["Value_y"]
                try:
                    idx = pd.RangeIndex(len(ratio))
                    slope = pd.Series(ratio.values).cov(idx) / pd.Series(idx).var()
                    out["sgaTrend"] = (slope is not None) and (slope < 0)
                except Exception:
                    pass

    return out
def make_criteria_with_labels(CATEGORIES):
    FIELD_ALIAS = {"trailingPE": "peRatio"}
    LABEL_MAP = {
        "earningsGrowth":   "Gewinnwachstum",
        "epsGrowth":        "EPS-Wachstum",
        "dividendYield":    "Dividendenrendite",
        "payoutRatio":      "Payout Ratio",
        "revenueGrowth":    "Umsatzwachstum",
        "peRatio":          "KGV",
        "debtToAssets":     "Debt/Assets",
        "freeCashFlowPerShare": "FCF/Aktie",
    }
    labeled = {}
    for cat, rules in CATEGORIES.items():
        augmented = []
        for item in rules:
            if len(item) == 2:
                field, rule = item
                optional = False
                label_text = LABEL_MAP.get(field, field)
            else:
                field, label_text_in, rule, optional = item
                label_text = label_text_in or LABEL_MAP.get(field, field)

            used_field = FIELD_ALIAS.get(field, field)
            augmented.append((used_field, label_text, rule, optional))
        labeled[cat] = augmented
    return labeled
# ==========================================================
# 7️⃣ Top10 Snapshot Storage (persistente Speicherung)
# ==========================================================

TOP10_SNAPSHOT_INDEX = os.getenv("TOP10_SNAPSHOT_INDEX", "top10_snapshots")


def ensure_top10_snapshot_index(es):
    """Creates the index for persistent Top10 snapshots if missing."""
    if es.indices.exists(index=TOP10_SNAPSHOT_INDEX):
        return

    es.indices.create(
        index=TOP10_SNAPSHOT_INDEX,
        body={
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "meta": {"type": "object", "enabled": True},
                    "filters": {"type": "object", "enabled": True},
                    "top10_by_category": {"type": "object", "enabled": True},
                }
            }
        },
    )


def save_top10_snapshot(es, name: str, payload: dict, snapshot_id: str | None = None) -> str:
    """
    Speichert einen Top10 Snapshot dauerhaft in ES.
    - name: sprechender Name
    - payload: {meta, top10_by_category, (optional) filters}
    - snapshot_id: optional (wenn du überschreiben willst), sonst wird neue ID erzeugt
    """
    now = datetime.now(timezone.utc).isoformat()

    doc = {
        "name": name.strip(),
        "created_at": now,
        "updated_at": now,
        "meta": payload.get("meta", {}),
        "filters": payload.get("filters", {}),
        "top10_by_category": payload.get("top10_by_category", {}),
    }

    if snapshot_id:
        # upsert
        es.update(
            index=TOP10_SNAPSHOT_INDEX,
            id=snapshot_id,
            body={"doc": doc, "doc_as_upsert": True},
            refresh=True,
        )
        return snapshot_id

    res = es.index(index=TOP10_SNAPSHOT_INDEX, document=doc, refresh=True)
    return res["_id"]


def list_top10_snapshots(es, limit: int = 200):
    """Listet Snapshots (neueste zuerst)."""
    resp = es.search(
        index=TOP10_SNAPSHOT_INDEX,
        body={
            "size": limit,
            "sort": [{"updated_at": {"order": "desc"}}],
            "_source": ["name", "updated_at", "meta", "filters"],
            "query": {"match_all": {}},
        },
    )
    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {}) or {}
        out.append({
            "id": h.get("_id"),
            "name": src.get("name", ""),
            "updated_at": src.get("updated_at"),
            "meta": src.get("meta", {}),
            "filters": src.get("filters", {}),
        })
    return out


def load_top10_snapshot(es, snapshot_id: str) -> Optional[dict]:
    """Lädt Snapshot by ID und gibt payload im gleichen Format zurück wie deine Session erwartet."""
    try:
        src = es.get(index=TOP10_SNAPSHOT_INDEX, id=snapshot_id)["_source"]
    except NotFoundError:
        return None

    return {
        "meta": src.get("meta", {}),
        "filters": src.get("filters", {}),
        "top10_by_category": src.get("top10_by_category", {}),
    }


def delete_top10_snapshot(es, snapshot_id: str) -> bool:
    try:
        es.delete(index=TOP10_SNAPSHOT_INDEX, id=snapshot_id, refresh=True)
        return True
    except NotFoundError:
        return False

