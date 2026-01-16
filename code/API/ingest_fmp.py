import os
import time
import random
import json
from datetime import datetime, UTC
from typing import Dict, List
from pathlib import Path

import requests
from elasticsearch import helpers
from dotenv import load_dotenv

# --- interne Helfer aus utils.py (noch zu erstellen) ---
from utils import (
    es_client,
    es_healthcheck,
    ensure_index,
    requests_session,
    random_user_agent,
)

# === 1️⃣ Setup & Konfiguration ===

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.local", override=False)
load_dotenv(BASE_DIR / ".env", override=False)

FMP_API_KEY = os.getenv("FMP_API_KEY")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "metrics_sp500")

if not FMP_API_KEY:
    raise RuntimeError("FMP_API_KEY fehlt. Bitte in .env.local setzen!")

es = es_client()
SESSION = requests_session()

DATA_DIR = Path(__file__).resolve().parents[0] / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "sp500_symbols.json"

# === 2️⃣ Metrik-Mapping (deine Felder → FMP Quote Felder) ===

METRIC_MAPPING: Dict[str, str] = {
    "peRatio": "pe",
    "bookValuePerShare": "bookValue",
    "dividendYield": "dividendYield",
    "priceToBook": "priceToBookRatio",
    "eps": "eps",
    "marketCap": "marketCap"
}


# === 3️⃣ Funktionen ===

def get_sp500_symbols(force_refresh: bool = False) -> List[str]:
    """Lädt S&P 500 Symbole (über SPY ETF als Free-Tier-Alternative, cached sie 24h)."""
    if CACHE_FILE.exists() and not force_refresh:
        age_hours = (datetime.now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600
        if age_hours < 24:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)

    url = f"https://financialmodelingprep.com/api/v3/etf-holdings/SPY?apikey={FMP_API_KEY}"
    r = SESSION.get(url, timeout=20, headers={"User-Agent": random_user_agent()})
    r.raise_for_status()
    data = r.json()

    symbols = [row["asset"] for row in data.get("holdings", []) if "asset" in row]
    with open(CACHE_FILE, "w") as f:
        json.dump(symbols, f, indent=2)

    print(f"✅ {len(symbols)} S&P 500 Symbole (aus SPY ETF) geladen und gecacht.")
    return symbols



def get_quote(symbol: str) -> Dict:
    """Lädt aktuelle Kennzahlen (Quote) für ein Symbol."""
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
    headers = {"User-Agent": random_user_agent()}
    r = SESSION.get(url, timeout=15, headers=headers)
    if r.status_code != 200:
        print(f"[WARN] {symbol}: HTTP {r.status_code}")
        return {}
    arr = r.json() or []
    return arr[0] if arr else {}


def build_doc(symbol: str, profile: Dict) -> Dict:
    """Erzeugt EIN Dokument pro Aktie mit allen Kennzahlen."""
    today = str(datetime.now(UTC).date())
    doc = {
        "_index": ES_INDEX,
        "_id": f"{symbol}|{today}",
        "_source": {
            "symbol": symbol,
            "date": today,
            "source": "FMP (Quote)",
            "ingested_at": datetime.now(UTC).isoformat(),
        },
    }

    for our_name, fmp_field in METRIC_MAPPING.items():
        value = profile.get(fmp_field)
        if isinstance(value, (int, float)):
            doc["_source"][our_name] = float(value)

    return doc


# === 4️⃣ Main-Pipeline ===

def run(batch_sleep: float = 7.0):
    """Hauptpipeline für FMP-Ingestion (S&P 500)."""
    print(es_healthcheck(es))
    ensure_index(es, ES_INDEX)

    symbols = get_sp500_symbols()
    random.shuffle(symbols)  # Anti-Bot
    print(f"Starte Ingestion für {len(symbols)} Symbole...")

    docs_buffer = []
    written = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            quote = get_quote(symbol)
            if not quote:
                continue

            doc = build_doc(symbol, quote)
            docs_buffer.append(doc)

            # Alle 100 Docs in Elasticsearch schreiben
            if len(docs_buffer) >= 100:
                helpers.bulk(es, docs_buffer)
                written += len(docs_buffer)
                docs_buffer.clear()
                print(f"[{i}/{len(symbols)}] {written} Dokumente gespeichert...")

        except Exception as e:
            print(f"[FEHLER] {symbol}: {e}")

        # Sleep mit Jitter (Anti-Bot)
        time.sleep(batch_sleep + random.uniform(0.3, 1.2))

    # Rest speichern
    if docs_buffer:
        helpers.bulk(es, docs_buffer)
        written += len(docs_buffer)

    print(f"✅ Fertig. Gesamt gespeichert: {written} Dokumente.")


# === 5️⃣ Einstiegspunkt ===

if __name__ == "__main__":
    run()
