# code/API/ingest_prices_sp2.py
# Lädt EOD-Preisdaten (adjClose) aus:
#   data/sp_data_2/total_sp_data_Prices/{SYMBOL}_eod_prices.json
# und schreibt sie in einen separaten Elasticsearch-Index (standard: "prices").

import os
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import helpers
from utils import es_client, es_healthcheck, ensure_index  # vorhanden in code/API/utils.py


# ===================== Config =====================
BASE_DIR    = Path(__file__).resolve().parent
PROJECTROOT = BASE_DIR.parents[1]
PRICE_DIR   = PROJECTROOT / "data" / "sp_data_2" / "total_sp_data_Prices"
PRICE_INDEX = os.getenv("ELASTICSEARCH_PRICE_INDEX", "prices")
BULK_FLUSH  = int(os.getenv("BULK_FLUSH", "2000"))
# Optional: Benchmarks (z.B. ^GSPC) explizit ignorieren, falls sie im Ordner liegen sollten
IGNORE_SYMBOLS = set(x.strip() for x in os.getenv("IGNORE_SYMBOLS", "^GSPC,^SPXEW").split(",") if x.strip())

es = es_client()


# ===================== Helpers =====================
def _read_json(path: Path) -> Optional[Any]:
    if not path.exists() or path.stat().st_size < 10:
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt or txt in ("[]", "{}"):
        return None
    return json.loads(txt)

def _to_iso_date(s: Any) -> Optional[str]:
    if not s:
        return None
    return str(s)[:10]  # erwartet YYYY-MM-DD

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def _extract_symbol_from_filename(p: Path) -> str:
    # "AAPL_eod_prices.json" -> "AAPL"
    name = p.name
    return name.split("_", 1)[0] if "_" in name else p.stem


def build_price_actions(file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Erwartete Struktur pro Datei:
    {
      "symbol": "AAPL",
      "historical": [
        {"date":"2026-01-09", "adjClose":..., "open":..., ...},
        ...
      ]
    }

    Wir speichern pro Datum 1 Dokument:
      _id = "{symbol}|{YYYY-MM-DD}|eod"
    """
    obj = _read_json(file_path)
    if not isinstance(obj, dict):
        return (_extract_symbol_from_filename(file_path), [])

    symbol = obj.get("symbol") or _extract_symbol_from_filename(file_path)
    if symbol in IGNORE_SYMBOLS:
        return (symbol, [])

    hist = obj.get("historical")
    if not isinstance(hist, list) or not hist:
        return (symbol, [])

    ingested_at = datetime.now(UTC).isoformat()
    actions: List[Dict[str, Any]] = []

    for row in hist:
        if not isinstance(row, dict):
            continue

        d = _to_iso_date(row.get("date"))
        if not d:
            continue

        adj = _safe_float(row.get("adjClose"))
        if adj is None:
            # adjClose ist dein primäres Feld – wenn nicht vorhanden, skip
            continue

        doc = {
            "symbol": symbol,
            "date": d,
            "source": "sp_data_2",
            "ingested_at": ingested_at,
            "adjClose": adj,

            # optional: weitere Felder, hilfreich für Charts/Debug/Qualität
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "close": _safe_float(row.get("close")),
            "volume": _safe_int(row.get("volume")),
            "vwap": _safe_float(row.get("vwap")),
        }
        # None-Felder entfernen
        doc = {k: v for k, v in doc.items() if v is not None}

        actions.append({
            "_op_type": "index",                    # idempotent
            "_index": PRICE_INDEX,
            "_id": f"{symbol}|{d}|eod",
            "_source": doc
        })

    return (symbol, actions)


def run():
    print(es_healthcheck(es))
    ensure_index(es, PRICE_INDEX)

    if not PRICE_DIR.exists():
        raise FileNotFoundError(f"PRICE_DIR nicht gefunden: {PRICE_DIR}")

    files = sorted(PRICE_DIR.glob("*_eod_prices.json"))
    if not files:
        print(f"⚠️ Keine *_eod_prices.json Dateien in: {PRICE_DIR}")
        return

    buffer: List[Dict[str, Any]] = []
    written = 0
    processed_files = 0

    for p in files:
        processed_files += 1
        symbol, actions = build_price_actions(p)

        if not actions:
            # Nicht als Fehler behandeln: manche Dateien können leer sein
            print(f"ℹ️  {symbol}: keine Preisdaten extrahiert ({p.name})")
            continue

        buffer.extend(actions)

        if len(buffer) >= BULK_FLUSH:
            helpers.bulk(es, buffer, raise_on_error=False)
            written += len(buffer)
            buffer.clear()
            print(f"[{processed_files}/{len(files)}] {written} Price-Dokumente gespeichert...")

    if buffer:
        helpers.bulk(es, buffer, raise_on_error=False)
        written += len(buffer)

    print(f"✅ Price-Ingest fertig. Gespeichert: {written} Dokumente in Index '{PRICE_INDEX}'.")


if __name__ == "__main__":
    run()
