# code/API/ingest_av.py
import os, json, time, random
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List
import requests
from elasticsearch import helpers
from utils import es_client, es_healthcheck, ensure_index  # <- vorhanden in API/utils.py

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "sp500_symbols.json"   # falls vorhanden
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "stocks")

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
assert API_KEY, "ALPHAVANTAGE_API_KEY fehlt (.env)!"

AV_BASE = "https://www.alphavantage.co/query"

es = es_client()

def load_symbols() -> List[str]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    # kleine Fallback-Liste
    return ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","PG","PFE","BAC","XOM","CVX","INTC","T","UNH","DIS"]

def av_get(params: Dict) -> Dict:
    params = {**params, "apikey": API_KEY}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # AV gibt bei Limits auch 200 zurück, aber mit "Note" / "Information"
    if "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Note") or data.get("Information"))
    return data

def fetch_overview(symbol: str) -> Dict:
    return av_get({"function": "OVERVIEW", "symbol": symbol})

def fetch_income(symbol: str) -> Dict:
    return av_get({"function": "INCOME_STATEMENT", "symbol": symbol})

def fetch_balance(symbol: str) -> Dict:
    return av_get({"function": "BALANCE_SHEET", "symbol": symbol})

def fetch_cashflow(symbol: str) -> Dict:
    return av_get({"function": "CASH_FLOW", "symbol": symbol})

def fetch_earnings(symbol: str) -> Dict:
    # reportedEPS je Quartal/Jahr (für EPS-/Earnings-Growth)
    return av_get({"function": "EARNINGS", "symbol": symbol})

# === Helper 
def _f(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def build_metrics(symbol: str) -> Dict:
    ov  = fetch_overview(symbol)
    inc = fetch_income(symbol)
    bal = fetch_balance(symbol)
    cfs = fetch_cashflow(symbol)
    ern = fetch_earnings(symbol)

    # ---- OVERVIEW (aktuell) -> direkte Mappings auf YF-Namen ----
    marketCap     = _f(ov.get("MarketCapitalization"))
    peRatio       = _f(ov.get("PERatio"))
    priceToBook   = _f(ov.get("PriceToBookRatio"))
    payoutRatio   = _f(ov.get("PayoutRatio"))
    dividendYield = _f(ov.get("DividendYield"))
    sector        = ov.get("Sector")
    industry      = ov.get("Industry")
    beta          = _f(ov.get("Beta"))
    pegRatio      = _f(ov.get("PEGRatio"))
    shares_out    = _f(ov.get("SharesOutstanding"))
    trailingAnnualDividendRate = _f(ov.get("DividendPerShare"))  # ~ yfinance 'trailingAnnualDividendRate'

    # ---- Quartals-Arrays (neu -> alt, Index 0 = jüngstes) ----
    q_inc = inc.get("quarterlyReports", []) or []
    q_bal = bal.get("quarterlyReports", []) or []
    q_cfs = cfs.get("quarterlyReports", []) or []

    def qv(arr, key):
        if not arr: return None
        return _f(arr[0].get(key))

    # === INCOME (Quartal) ===
    revenue    = qv(q_inc, "totalRevenue")
    net_income = qv(q_inc, "netIncome")
    profitMargin = (net_income / revenue) if isinstance(net_income, float) and isinstance(revenue, float) and revenue else None

    # YoY Revenue Growth (Vorjahresquartal ~ Index 4)
    revenueGrowth = None
    try:
        if len(q_inc) >= 5:
            rev_latest = _f(q_inc[0].get("totalRevenue"))
            rev_prev_y = _f(q_inc[4].get("totalRevenue"))
            if isinstance(rev_latest, float) and isinstance(rev_prev_y, float) and rev_prev_y:
                revenueGrowth = (rev_latest - rev_prev_y) / abs(rev_prev_y)
    except Exception:
        pass

    # === SG&A TREND (rückläufige Quote) ===
    def _sga_val(rec):
        # AV nutzt teils unterschiedliche Keys – versuche mehrere Varianten.
        for k in (
            "sellingGeneralAdministrative",
            "sellingGeneralAndAdministrative",
            "sellingGeneralAndAdministration",
            "sellingGeneralAndAdmin",
        ):
            v = rec.get(k)
            if v is not None:
                return _f(v)
        return None

    sgaTrend = None
    try:
        # Baue die letzten bis zu 4 SG&A-Quoten (Quartal 0..3)
        ratios = []
        for i in range(min(4, len(q_inc))):
            sga_i = _sga_val(q_inc[i])
            rev_i = _f(q_inc[i].get("totalRevenue"))
            if isinstance(sga_i, float) and isinstance(rev_i, float) and rev_i:
                ratios.append(sga_i / rev_i)

        if len(ratios) >= 3:
            # Kriterium: klar abnehmender Verlauf ODER jüngste Quote unter dem Durchschnitt der Vorperioden
            monotonic_down = ratios[0] <= ratios[1] <= (ratios[2] if len(ratios) > 2 else ratios[1])
            last_below_mean_prev = ratios[0] < (sum(ratios[1:]) / len(ratios[1:]))
            sgaTrend = bool(monotonic_down or last_below_mean_prev)
        elif len(ratios) == 2:
            sgaTrend = bool(ratios[0] <= ratios[1])
        else:
            sgaTrend = None
    except Exception:
        sgaTrend = None

    # === BALANCE (Quartal) ===
    totalAssets  = qv(q_bal, "totalAssets")
    total_equity = qv(q_bal, "totalShareholderEquity")
    totalCash    = qv(q_bal, "cashAndCashEquivalentsAtCarryingValue") or qv(q_bal, "cashAndCashEquivalents")
    cur_assets   = qv(q_bal, "totalCurrentAssets")
    cur_liab     = qv(q_bal, "totalCurrentLiabilities")
    inventory    = qv(q_bal, "inventory")

    # totalDebt: bevorzugt long + short (Fallback: shortLongTermDebtTotal)
    long_debt  = qv(q_bal, "longTermDebt")
    short_debt = qv(q_bal, "shortTermDebt")
    sld_total  = qv(q_bal, "shortLongTermDebtTotal")
    if isinstance(long_debt, float) or isinstance(short_debt, float):
        totalDebt = (long_debt or 0.0) + (short_debt or 0.0)
    else:
        totalDebt = sld_total  # kann None sein

    currentRatio = (cur_assets / cur_liab) if isinstance(cur_assets, float) and isinstance(cur_liab, float) and cur_liab else None
    quickRatio   = None
    if isinstance(cur_assets, float) and isinstance(inventory, float) and isinstance(cur_liab, float) and cur_liab:
        quickRatio = (cur_assets - inventory) / cur_liab

    debtToEquity = (totalDebt / total_equity) if isinstance(totalDebt, float) and isinstance(total_equity, float) and total_equity else None
    debtToAssets = (totalDebt / totalAssets) if isinstance(totalDebt, float) and isinstance(totalAssets, float) and totalAssets else None  # <-- NEU

    # === CASHFLOW (Quartal) -> Free Cash Flow (OCF - CapEx) ===
    ocf   = qv(q_cfs, "operatingCashflow") or qv(q_cfs, "operatingCashFlow")
    capex = qv(q_cfs, "capitalExpenditures") or qv(q_cfs, "capitalExpenditure")
    freeCashflow = (ocf - abs(capex)) if isinstance(ocf, float) and isinstance(capex, float) else None
    cashPerShare = (totalCash / shares_out) if isinstance(totalCash, float) and isinstance(shares_out, float) and shares_out else None
    freeCashFlowPerShare = (freeCashflow / shares_out) if isinstance(freeCashflow, float) and isinstance(shares_out, float) and shares_out else None

    # Abgeleitet wie in ingest_yf
    cashToDebt  = (totalCash / totalDebt) if isinstance(totalCash, float) and isinstance(totalDebt, float) and totalDebt else None
    equityRatio = (total_equity / totalAssets) if isinstance(total_equity, float) and isinstance(totalAssets, float) and totalAssets else None
    fcfMargin   = (freeCashflow / revenue) if isinstance(freeCashflow, float) and isinstance(revenue, float) and revenue else None

    # === EARNINGS (Quartal) -> epsGrowth/earningsGrowth YoY ===
    epsGrowth = earningsGrowth = None
    try:
        q_ern = ern.get("quarterlyEarnings", []) or []
        if len(q_ern) >= 5:
            eps_latest = _f(q_ern[0].get("reportedEPS"))
            eps_prev_y = _f(q_ern[4].get("reportedEPS"))
            if isinstance(eps_latest, float) and isinstance(eps_prev_y, float) and eps_prev_y:
                epsGrowth = (eps_latest - eps_prev_y) / abs(eps_prev_y)
                earningsGrowth = epsGrowth  # Alias wie bei yfinance
    except Exception:
        pass

    # Buchwert/Aktie (falls nicht direkt geliefert)
    bookValuePerShare = None
    if isinstance(total_equity, float) and isinstance(shares_out, float) and shares_out:
        bookValuePerShare = total_equity / shares_out

    # trailing EPS (wie yfinance 'trailingEps') hat AV nicht 1:1 → optional None
    eps = None

    # === Metriken (inkl. Aliasse zu deinen CATEGORIES) ===
    metrics = {
        "marketCap": marketCap,
        "peRatio": peRatio,
        "trailingPE": peRatio,                  # <-- Alias für deine Regeln
        "priceToBook": priceToBook,
        "dividendYield": dividendYield,
        "payoutRatio": payoutRatio,
        "sector": sector,
        "industry": industry,
        "beta": beta,
        "pegRatio": pegRatio,
        "trailingAnnualDividendRate": trailingAnnualDividendRate,

        "revenue": revenue,
        "profitMargin": profitMargin,
        "revenueGrowth": revenueGrowth,

        "totalAssets": totalAssets,
        "totalDebt": totalDebt,
        "totalCash": totalCash,
        "totalStockholderEquity": total_equity,
        "sharesOutstanding": shares_out,

        "currentRatio": currentRatio,
        "quickRatio": quickRatio,
        "debtToEquity": debtToEquity,
        "debtToAssets": debtToAssets,           # <-- NEU

        "freeCashflow": freeCashflow,
        "freeCashFlow": freeCashflow,           # <-- Alias exakt wie in CATEGORIES
        "cashPerShare": cashPerShare,
        "freeCashFlowPerShare": freeCashFlowPerShare,

        # abgeleitet wie yfinance
        "cashToDebt": cashToDebt,
        "equityRatio": equityRatio,
        "fcfMargin": fcfMargin,
        "earningsGrowth": earningsGrowth,
        "epsGrowth": epsGrowth,

        "bookValuePerShare": bookValuePerShare,
        "eps": eps,  # optional None

        "sgaTrend": sgaTrend,                   # <-- NEU (Turnaround-Signal)
    }

    # Nur nicht-None Felder zurückgeben
    return {k: v for k, v in metrics.items() if v is not None}
