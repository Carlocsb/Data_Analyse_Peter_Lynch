# code/API/ingest_sp_benchmarks.py
import os
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
from elasticsearch import helpers
from utils import es_client, es_healthcheck, ensure_index  # vorhanden in code/API/utils.py

# === Pfade & Config ===
BASE_DIR    = Path(__file__).resolve().parent          # .../code/API
PROJECTROOT = BASE_DIR.parents[1]                      # .../ (Projektwurzel)

SP2_DIR     = PROJECTROOT / "data" / "sp_data_2"       # neuer Ordner (auf Höhe data/...)
BENCH_INDEX = os.getenv("ELASTICSEARCH_BENCH_INDEX", "benchmarks")  # separater Index

# Welche Benchmark-Dateien sollen ingestiert werden?
BENCH_FILES = [
    "^GSPC_eod_prices.json",
    
]

es = es_client()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size < 10:
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt or txt in ("[]", "{}"):
        return None
    return json.loads(txt)


def _to_iso_date(s: Any) -> Optional[str]:
    if not s:
        return None
    ss = str(s)[:10]
    try:
        # "YYYY-MM-DD"
        return datetime.fromisoformat(ss).date().isoformat()
    except Exception:
        return None


def build_benchmark_actions(file_path: Path) -> List[Dict[str, Any]]:
    """
    Erwartete Struktur:
    {
      "symbol": "^GSPC",
      "historical": [
        {"date":"2026-01-09", "adjClose":..., "open":..., ...},
        ...
      ]
    }
    """
    payload = _read_json(file_path)
    if not payload or not isinstance(payload, dict):
        return []

    symbol = payload.get("symbol") or file_path.stem.replace("_eod_prices", "").replace("_autoadjusted", "")
    hist = payload.get("historical")

    if not isinstance(hist, list) or not hist:
        return []

    actions: List[Dict[str, Any]] = []
    ingested_at = datetime.now(UTC).isoformat()

    for row in hist:
        if not isinstance(row, dict):
            continue

        d = _to_iso_date(row.get("date"))
        if not d:
            continue

        adj = row.get("adjClose")
        if adj is None:
            # User-Requirement: adjClose ist das relevante Feld
            continue

        doc = {
            "symbol": symbol,
            "date": d,
            "source": "local_sp_data_2",
            "ingested_at": ingested_at,

            # wichtigstes Feld:
            "adjClose": float(adj),

            # optional mitnehmen (hilfreich für Charts/Checks):
            "open": float(row["open"]) if row.get("open") is not None else None,
            "high": float(row["high"]) if row.get("high") is not None else None,
            "low":  float(row["low"])  if row.get("low")  is not None else None,
            "close": float(row["close"]) if row.get("close") is not None else None,
            "volume": int(row["volume"]) if row.get("volume") is not None else None,
            "vwap": float(row["vwap"]) if row.get("vwap") is not None else None,
            "change": float(row["change"]) if row.get("change") is not None else None,
            "changePercent": float(row["changePercent"]) if row.get("changePercent") is not None else None,
        }

        # None-Felder entfernen (sauberer ES-Doc)
        doc = {k: v for k, v in doc.items() if v is not None}

        actions.append({
            "_op_type": "index",  # idempotent: bei erneutem Lauf wird aktualisiert
            "_index": BENCH_INDEX,
            "_id": f"{symbol}|{d}|eod",
            "_source": doc
        })

    return actions


def run():
    print(es_healthcheck(es))
    ensure_index(es, BENCH_INDEX)

    if not SP2_DIR.exists():
        raise FileNotFoundError(f"sp_data_2 Ordner nicht gefunden: {SP2_DIR}")

    buffer: List[Dict[str, Any]] = []
    written = 0

    for fname in BENCH_FILES:
        p = SP2_DIR / fname
        if not p.exists():
            print(f"⚠️ Datei fehlt: {p.name} — übersprungen.")
            continue

        actions = build_benchmark_actions(p)
        if not actions:
            print(f"⚠️ Keine Daten extrahiert aus {p.name}.")
            continue

        buffer.extend(actions)

    if buffer:
        helpers.bulk(es, buffer, raise_on_error=False)
        written = len(buffer)

    print(f"✅ Benchmark-Ingest fertig. Gespeichert: {written} Dokumente in Index '{BENCH_INDEX}'.")


if __name__ == "__main__":
    run()
