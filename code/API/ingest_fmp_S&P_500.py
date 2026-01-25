# code/API/ingest_sp_benchmarks.py
import os
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
from elasticsearch import helpers
from utils import es_client, es_healthcheck, ensure_index  # vorhanden in code/API/utils.py

# === Pfade & Config ===
BASE_DIR = Path(__file__).resolve().parent          # .../code/API
PROJECTROOT = BASE_DIR.parents[1]                   # .../ (Projektwurzel)

SP2_DIR = PROJECTROOT / "data" / "sp_data_2"        # neuer Ordner (auf Höhe data/...)
BENCH_INDEX = os.getenv("ELASTICSEARCH_BENCH_INDEX", "benchmarks")  # separater Index

# Welche Benchmark-Dateien sollen ingestiert werden?
BENCH_FILES = [
    "^GSPC_eod_prices.json",        # FMP-Format: {"symbol":..., "historical":[...]}
    "^SPXEW_autoadjusted.json",     # Date-Key-Format: {"YYYY-MM-DD": {"Open":..., ...}, ...}
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
        return datetime.fromisoformat(ss).date().isoformat()
    except Exception:
        return None


def _dataset_from_filename(file_path: Path) -> str:
    name = file_path.name.lower()
    if "autoadjusted" in name:
        return "autoadjusted"
    if "eod_prices" in name:
        return "eod_prices"
    return "unknown"


def build_benchmark_actions(file_path: Path) -> List[Dict[str, Any]]:
    """
    Unterstützt zwei Strukturen:

    A) FMP-Style:
    {
      "symbol": "^GSPC",
      "historical": [
        {"date":"2026-01-09", "adjClose":..., "open":..., ...},
        ...
      ]
    }

    B) Date-Key-Style (z.B. yfinance/pandas export):
    {
      "2009-01-09": {"Open":..., "High":..., "Low":..., "Close":..., "Volume":...},
      "2009-01-12": {...},
      ...
    }
    """
    payload = _read_json(file_path)
    if not payload or not isinstance(payload, dict):
        return []

    dataset = _dataset_from_filename(file_path)
    ingested_at = datetime.now(UTC).isoformat()

    # Symbol: aus payload (wenn vorhanden) sonst aus Dateiname ableiten
    symbol = payload.get("symbol") or file_path.stem.replace("_eod_prices", "").replace("_autoadjusted", "")

    actions: List[Dict[str, Any]] = []

    # -----------------------------
    # Fall A: FMP-Style (historical)
    # -----------------------------
    hist = payload.get("historical")
    if isinstance(hist, list) and hist:
        for row in hist:
            if not isinstance(row, dict):
                continue

            d = _to_iso_date(row.get("date"))
            if not d:
                continue

            # Primary + Fallback
            adj = row.get("adjClose")
            close = row.get("close")

            # Wenn adjClose fehlt, nutze close als Fallback (falls vorhanden)
            if adj is None and close is None:
                continue

            doc = {
                "symbol": symbol,
                "date": d,
                "dataset": dataset,
                "source": "local_sp_data_2",
                "ingested_at": ingested_at,

                # wichtigstes Feld (Fallback auf close)
                "adjClose": float(adj if adj is not None else close),

                # close zusätzlich speichern
                "close": float(close) if close is not None else None,

                # optional mitnehmen:
                "open": float(row["open"]) if row.get("open") is not None else None,
                "high": float(row["high"]) if row.get("high") is not None else None,
                "low":  float(row["low"])  if row.get("low")  is not None else None,
                "volume": int(row["volume"]) if row.get("volume") is not None else None,
                "vwap": float(row["vwap"]) if row.get("vwap") is not None else None,
                "change": float(row["change"]) if row.get("change") is not None else None,
                "changePercent": float(row["changePercent"]) if row.get("changePercent") is not None else None,
            }

            doc = {k: v for k, v in doc.items() if v is not None}

            actions.append({
                "_op_type": "index",
                "_index": BENCH_INDEX,
                "_id": f"{symbol}|{d}|{dataset}",
                "_source": doc
            })

        return actions

    # ---------------------------------------
    # Fall B: Date-Key-Style (YYYY-MM-DD keys)
    # ---------------------------------------
    # payload enthält KEIN "historical" => wir interpretieren die Keys als Datum
    for date_key, row in payload.items():
        d = _to_iso_date(date_key)
        if not d or not isinstance(row, dict):
            continue

        # yfinance/pandas keys: Open/High/Low/Close/Volume (Großschreibung!)
        o = row.get("Open")
        h = row.get("High")
        l = row.get("Low")
        c = row.get("Close")
        v = row.get("Volume")

        if c is None:
            continue

        doc = {
            "symbol": symbol,
            "date": d,
            "dataset": dataset,
            "source": "local_sp_data_2",
            "ingested_at": ingested_at,

            # Für autoadjusted-Dateien ist Close i.d.R. bereits adjusted -> als adjClose spiegeln
            "adjClose": float(c),
            "close": float(c),

            "open": float(o) if o is not None else None,
            "high": float(h) if h is not None else None,
            "low":  float(l) if l is not None else None,
            "volume": int(v) if v is not None else None,
        }

        doc = {k: v for k, v in doc.items() if v is not None}

        actions.append({
            "_op_type": "index",
            "_index": BENCH_INDEX,
            "_id": f"{symbol}|{d}|{dataset}",
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
