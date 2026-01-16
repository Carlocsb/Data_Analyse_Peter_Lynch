import os
import json
import time
import random
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List
import yfinance as yf
from elasticsearch import helpers
from dotenv import load_dotenv
from utils import es_client, es_healthcheck, ensure_index

# === 1️⃣ Setup ===
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(BASE_DIR / ".env.local", override=False)

ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "stocks")
es = es_client()

DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "sp500_symbols.json"

# === 2️⃣ Symbol-Quelle ===
def load_symbols() -> List[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _fallback_symbols() -> List[str]:
        return [
            "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","JNJ","PG",
            "V","MA","HD","DIS","PFE","NFLX","KO","PEP","XOM","CSCO","BAC",
            "CVX","NKE","ORCL","ABBV","INTC","ADBE","T","WMT","UNH"
        ]

    # 1) Cache lesen
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                symbols = json.load(f)
            if isinstance(symbols, list) and len(symbols) > 0:
                print(f"✅ {len(symbols)} Symbole aus Cache geladen.")
                return symbols
            else:
                print("⚠️ Cache ist leer. Versuche Neuaufbau…")
        except Exception as e:
            print(f"⚠️ Cache defekt ({e}). Baue neu auf…")

    # 2) Neuaufbau via yfinance (S&P 500)
    try:
        syms = yf.tickers_sp500()
        syms = [s.strip().upper() for s in syms if s and isinstance(s, str)]
        if syms:
            with open(CACHE_FILE, "w") as f:
                json.dump(syms, f)
            print(f"✅ S&P-500 neu geladen: {len(syms)} Symbole. Cache aktualisiert.")
            return syms
    except Exception as e:
        print(f"⚠️ Konnte S&P-500 nicht laden ({e}). Fallback auf Test-Symbole.")

    # 3) Fallback
    test_syms = _fallback_symbols()
    print(f"⚠️ Keine Cache-Datei / kein Download möglich – nutze {len(test_syms)} Test-Symbole.")
    return test_syms

# === 3️⃣ Metriken ===
# Nur informativ – wir lesen dynamisch aus FIELD_MAP; diese Liste muss nicht 1:1 genutzt werden.
METRIC_FIELDS = [
    "trailingPE", "priceToBook", "trailingEps", "dividendYield", "marketCap",
    "bookValue", "freeCashflow", "totalDebt", "totalRevenue",
    "revenueGrowth", "profitMargins", "pegRatio", "sector", "industry", "beta",
    "trailingAnnualDividendRate", "debtToEquity", "quickRatio", "currentRatio",
    "payoutRatio", "totalCashPerShare", "earningsGrowth", "totalAssets",
    # NEW ↓
    "totalCash", "sharesOutstanding", "totalStockholderEquity",
]

# yfinance.info → interne Feldnamen
FIELD_MAP = {
    "trailingPE": "peRatio",
    "priceToBook": "priceToBook",
    "trailingEps": "eps",
    "dividendYield": "dividendYield",
    "marketCap": "marketCap",
    "bookValue": "bookValuePerShare",          # yfinance 'bookValue' ist pro Aktie
    "freeCashflow": "freeCashflow",
    "totalDebt": "totalDebt",
    "totalRevenue": "revenue",
    "revenueGrowth": "revenueGrowth",
    "profitMargins": "profitMargin",
    "pegRatio": "pegRatio",
    "sector": "sector",
    "industry": "industry",
    "beta": "beta",
    "trailingAnnualDividendRate": "trailingAnnualDividendRate",
    "debtToEquity": "debtToEquity",
    "quickRatio": "quickRatio",
    "currentRatio": "currentRatio",
    "payoutRatio": "payoutRatio",
    "totalCashPerShare": "cashPerShare",
    "earningsGrowth": "earningsGrowth",
    "totalAssets": "totalAssets",
    # NEW ↓ Rohwerte für abgeleitete Kennzahlen
    "totalCash": "totalCash",
    "sharesOutstanding": "sharesOutstanding",
    "totalStockholderEquity": "totalStockholderEquity",
}

# === 4️⃣ Daten laden ===
def get_metrics(symbol: str) -> Dict:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    metrics: Dict[str, float | str] = {}

    # Rohfelder mappen
    for yf_field, our_field in FIELD_MAP.items():
        val = info.get(yf_field)
        # yfinance liefert oft None, ints, floats, teils strings
        if isinstance(val, (int, float)):
            metrics[our_field] = float(val)
        elif isinstance(val, str):
            metrics[our_field] = val

    # === 4a) Abgeleitete Kennzahlen (für Lynch-Kategorien) ===
    # cashToDebt: Cash >= 50 % der Schulden
    total_cash = metrics.get("totalCash")
    total_debt = metrics.get("totalDebt")
    if isinstance(total_cash, float) and isinstance(total_debt, float) and total_debt not in (0.0, None):
        metrics["cashToDebt"] = total_cash / total_debt

    # equityRatio: Eigenkapitalquote = Equity / Assets
    total_equity = metrics.get("totalStockholderEquity")
    total_assets = metrics.get("totalAssets")
    if isinstance(total_equity, float) and isinstance(total_assets, float) and total_assets not in (0.0, None):
        metrics["equityRatio"] = total_equity / total_assets

    # freeCashFlowPerShare
    fcf = metrics.get("freeCashflow")
    shares = metrics.get("sharesOutstanding")
    if isinstance(fcf, float) and isinstance(shares, float) and shares not in (0.0, None):
        metrics["freeCashFlowPerShare"] = fcf / shares

    # fcfMargin: FCF / Revenue
    revenue = metrics.get("revenue")
    if isinstance(fcf, float) and isinstance(revenue, float) and revenue not in (0.0, None):
        metrics["fcfMargin"] = fcf / revenue

    # Harmonisierung/Backfills für Scoring:
    # trailingPE-Fallback (falls peRatio leer): yfinance nutzt trailingPE -> oben schon 'peRatio'
    if "peRatio" not in metrics and isinstance(info.get("trailingPE"), (int, float)):
        metrics["peRatio"] = float(info["trailingPE"])

    # earningsGrowth als epsGrowth-Alias falls benötigt
    if "epsGrowth" not in metrics and isinstance(metrics.get("earningsGrowth"), float):
        metrics["epsGrowth"] = metrics["earningsGrowth"]

    # --- Aliasse für Kategorien ---
    if "peRatio" in metrics and "trailingPE" not in metrics:
        metrics["trailingPE"] = metrics["peRatio"]

    if "freeCashflow" in metrics and "freeCashFlow" not in metrics:
        metrics["freeCashFlow"] = metrics["freeCashflow"]

    # --- Debt/Assets (für mehrere Kategorien) ---
    if isinstance(total_debt, float) and isinstance(total_assets, float) and total_assets:
        metrics["debtToAssets"] = total_debt / total_assets

    # --- Sektor normalisieren (für Cyclicals) ---
    def _norm_sector(s: str) -> str:
        m = {
            "consumer cyclical": "cyclical",
            "consumer discretionary": "cyclical",
            "industrials": "cyclical",
            "materials": "basic materials",
            "energy": "energy",
            "automotive": "auto",
            "automobiles & components": "auto",
        }
        return m.get(str(s).strip().lower(), str(s).strip().lower())
    

    if isinstance(metrics.get("sector"), str):
        metrics["sector"] = _norm_sector(metrics["sector"])
        
    return metrics

# === 5️⃣ Dokument aufbauen ===
def build_doc(symbol: str, metrics: Dict) -> Dict:
    today = str(datetime.now(UTC).date())
    return {
        "_index": ES_INDEX,
        "_id": f"{symbol}|{today}",
        "_source": {
            "symbol": symbol,
            "date": today,
            "source": "yfinance",
            "ingested_at": datetime.now(UTC).isoformat(),
            **metrics
        },
    }

# === 6️⃣ Pipeline ===
def run(batch_sleep: float = 2.5):
    print(es_healthcheck(es))
    ensure_index(es, ES_INDEX)

    symbols = load_symbols()
    random.shuffle(symbols)

    docs_buffer = []
    written = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            metrics = get_metrics(symbol)
            if not metrics:
                continue

            doc = build_doc(symbol, metrics)
            docs_buffer.append(doc)

            if len(docs_buffer) >= 25:
                helpers.bulk(es, docs_buffer)
                written += len(docs_buffer)
                docs_buffer.clear()
                print(f"[{i}/{len(symbols)}] {written} Dokumente gespeichert...")

        except Exception as e:
            print(f"[FEHLER] {symbol}: {e}")

        time.sleep(batch_sleep + random.uniform(0.4, 0.8))

    if docs_buffer:
        helpers.bulk(es, docs_buffer)
        written += len(docs_buffer)

    print(f"✅ Fertig. Gesamt gespeichert: {written} Dokumente.")

# === 7️⃣ Einstiegspunkt ===
if __name__ == "__main__":
    run()