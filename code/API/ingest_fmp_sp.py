# code/API/ingest_fmp_sp.py
import os, json, random, math
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Any, Iterable, Tuple
from elasticsearch import helpers
from utils import es_client, es_healthcheck, ensure_index  # vorhanden in code/API/utils.py

# === Pfade & Config ===
BASE_DIR    = Path(__file__).resolve().parent          # .../code/API
PROJECTROOT = BASE_DIR.parents[1]                      # .../ (Projektwurzel)
FMP_DIR     = PROJECTROOT / "data" / "sp_data" / "total_sp_data"  # Ordner mit den FMP-Dateien
ES_INDEX    = os.getenv("ELASTICSEARCH_INDEX", "stocks")
STRICT_MODE = os.getenv("STRICT_REQUIRED", "0") == "1"  # 1 = streng, 0 = aufnehmen + warnen

es = es_client()

# === Anforderungen definieren ===
REQUIRED_FILES = ["Profile", "IncomeStatement", "BalanceSheet", "CashflowStatement", "KeyMetrics", "Ratios"]

# Alle Felder, die deine CATEGORIES verwenden (strikt)
REQUIRED_FIELDS = {
    # Slow Growers
    "earningsGrowth", "dividendYield", "payoutRatio", "revenueGrowth", "trailingPE", "debtToAssets",
    # Stalwarts
    "marketCap", "freeCashFlow", "debtToAssets",
    # Fast Growers
    "pegRatio", "priceToBook",
    # Cyclicals
    "sector", "epsGrowth", "freeCashFlowPerShare",
    # Turnarounds
    "cashToDebt", "equityRatio", "fcfMargin", "currentRatio",
    # Asset Plays
    "bookValuePerShare", "cashPerShare",
    # Von dir generell genutzt
    "debtToEquity",
}

# ===================== kleine Helfer =====================
def _json_has_content(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 10:
        return False
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt or txt in ("[]", "{}"):
            return False
        json.loads(txt)
        return True
    except Exception:
        return False

def _has_all_required_files(sym: str) -> bool:
    for name in REQUIRED_FILES:
        if not _json_has_content(FMP_DIR / f"{sym}_{name}.json"):
            return False
    return True

def _missing_required_fields(metrics: dict) -> List[str]:
    return sorted(k for k in REQUIRED_FIELDS if k not in metrics or metrics[k] is None)

def _f(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def _read_json(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt: return None
    try:
        return json.loads(txt)
    except Exception:
        rows = []
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln: continue
            try: rows.append(json.loads(ln))
            except: pass
        return rows or None

def _latest_row(seq):
    if isinstance(seq, list) and seq:
        return seq[0] if isinstance(seq[0], dict) else {}
    return seq if isinstance(seq, dict) else {}

def _discover_symbols(folder: Path) -> List[str]:
    syms = set()
    for p in folder.glob("*_*.json"):
        name = p.name
        if "_" in name:
            syms.add(name.split("_", 1)[0])
    return sorted(syms)

def _to_date(s: Any):
    if not s: return None
    ss = str(s)[:10]
    try:
        return datetime.fromisoformat(ss).date()
    except Exception:
        try:
            from datetime import datetime as _dt
            return _dt.strptime(ss, "%Y-%m-%d").date()
        except Exception:
            return None

def _rows_with_date(obj, date_key="date") -> Iterable[Tuple[datetime.date, Dict[str, Any]]]:
    if isinstance(obj, list):
        for r in obj:
            if not isinstance(r, dict):
                continue
            d = _to_date(r.get(date_key))
            if d:
                yield d, r
    elif isinstance(obj, dict):
        d = _to_date(obj.get(date_key))
        if d:
            yield d, obj

def _merge_dict(dst: dict, src: dict, prefer_existing: bool = True) -> dict:
    for k, v in src.items():
        if v is None:
            continue
        if prefer_existing and (k in dst) and (dst[k] is not None):
            continue
        dst[k] = v
    return dst
def _normalize_numeric_fields(d: dict) -> dict:
    num_keys = [
        "marketCap","peRatio","priceToBook","dividendYield","payoutRatio",
        "revenueGrowth","earningsGrowth","epsGrowth","profitMargin","fcfMargin",
        "currentRatio","quickRatio","debtToAssets","freeCashFlow","freeCashflow",
        "freeCashFlowPerShare","cashPerShare","bookValuePerShare","totalDebt",
        "totalAssets","revenue","eps","beta","pegRatio","debtToEquity",
        "cashToDebt","equityRatio",
        # rohe FMP-Ratio-Felder, die wir später mappen
        "priceEarningsRatioTTM","priceEarningsRatio",
        "priceToBookRatio","pbRatio",
        "dividendYieldTTM",
        "priceEarningsToGrowthRatio","priceEarningsToGrowthRatioTTM",
        "debtToEquityTTM","debtEquityRatio",
        "currentRatioTTM","quickRatioTTM",
    ]
    for k in num_keys:
        if k in d:
            d[k] = _f(d[k])
    return d


# ===================== Heute-Dokument (aktuelle Kennzahlen) =====================
def build_metrics_fmp(symbol: str, base_dir: Path) -> Dict[str, Any]:
    files = {
        "Profile":           base_dir / f"{symbol}_Profile.json",
        "IncomeStatement":   base_dir / f"{symbol}_IncomeStatement.json",
        "BalanceSheet":      base_dir / f"{symbol}_BalanceSheet.json",
        "CashflowStatement": base_dir / f"{symbol}_CashflowStatement.json",
        "KeyMetrics":        base_dir / f"{symbol}_KeyMetrics.json",
        "Ratios":            base_dir / f"{symbol}_Ratios.json",
    }
    data = {k: (_read_json(p) if p.exists() else None) for k, p in files.items()}

    empty_parts = [k for k, v in data.items() if not v or v in ([], {}, None)]
    if empty_parts:
        print(f"⚠️ {symbol}: leere/fehlende Dateien → {', '.join(empty_parts)}")

    prof = data.get("Profile")
    if isinstance(prof, list) and prof: prof = prof[0]
    if not isinstance(prof, dict): prof = {}

    inc_list = data.get("IncomeStatement") if isinstance(data.get("IncomeStatement"), list) else []
    bal_list = data.get("BalanceSheet") if isinstance(data.get("BalanceSheet"), list) else []
    cfs_list = data.get("CashflowStatement") if isinstance(data.get("CashflowStatement"), list) else []
    kms_list = data.get("KeyMetrics") if isinstance(data.get("KeyMetrics"), list) else []
    rat_list = data.get("Ratios") if isinstance(data.get("Ratios"), list) else []

    inc  = _latest_row(inc_list)
    bal  = _latest_row(bal_list)
    cfs  = _latest_row(cfs_list)
    kms  = _latest_row(kms_list)
    rat  = _latest_row(rat_list)

    def pick(*keys):
        for k in keys:
            v = None
            # TTM/Key Felder inkl. payoutRatio
            if k in ("peRatio","priceToBook","dividendYield","pegRatio",
                     "freeCashFlowPerShare","bookValuePerShare","cashPerShare",
                     "debtToEquity","currentRatio","quickRatio","payoutRatio"):
                v = _f(kms.get(k+"TTM") if kms else None)
                if v is None and rat:
                    m = {
                        "peRatio": ("priceEarningsRatioTTM","priceEarningsRatio"),
                        "priceToBook": ("priceToBookRatio","pbRatio"),
                        "dividendYield": ("dividendYieldTTM","dividendYield"),
                        "pegRatio": ("pegRatioTTM","pegRatio"),
                        "debtToEquity": ("debtToEquityTTM","debtEquityRatio"),
                        "currentRatio": ("currentRatioTTM","currentRatio"),
                        "quickRatio": ("quickRatioTTM","quickRatio"),
                        "payoutRatio": ("payoutRatioTTM","payoutRatio"),
                    }.get(k, ())
                    for alt in m:
                        v = _f(rat.get(alt))
                        if v is not None: break
                if v is None and kms:
                    v = _f(kms.get(k) or kms.get({"priceToBook":"pbRatio"}.get(k,"")))
            else:
                v = _f(kms.get(k) if kms else None)
                if v is None and rat:
                    v = _f(rat.get(k))
                if v is None and prof:
                    v = _f(prof.get(k))
            if v is not None:
                return v
        return None

    def _norm_sector(s: str) -> str:
        m = {
            "consumer discretionary": "cyclical",
            "consumer cyclical": "cyclical",
            "industrials": "cyclical",
            "materials": "basic materials",
            "energy": "energy",
            "automobiles & components": "auto",
            "automotive": "auto",
        }
        if not isinstance(s, str): return s
        key = s.strip().lower()
        return m.get(key, key)

    marketCap   = _f(prof.get("mktCap") or prof.get("marketCap"))
    beta        = _f(prof.get("beta"))
    sector      = _norm_sector(prof.get("sector"))
    industry    = prof.get("industry")

    revenue     = _f(inc.get("revenue") or inc.get("totalRevenue"))
    net_income  = _f(inc.get("netIncome"))
    eps         = _f(inc.get("eps"))
    profitMargin = (net_income / revenue) if isinstance(net_income, float) and isinstance(revenue, float) and revenue else None

    revenueGrowth = None
    if len(inc_list) >= 2:
        r0 = _f(inc_list[0].get("revenue") or inc_list[0].get("totalRevenue"))
        r1 = _f(inc_list[1].get("revenue") or inc_list[1].get("totalRevenue"))
        if isinstance(r0, float) and isinstance(r1, float) and r1:
            revenueGrowth = (r0 - r1) / abs(r1)

    totalAssets   = _f(bal.get("totalAssets"))
    equity        = _f(bal.get("totalStockholdersEquity") or bal.get("totalStockholderEquity"))
    cash_eq       = _f(bal.get("cashAndCashEquivalents"))
    sti           = _f(bal.get("shortTermInvestments"))
    totalCash     = ((cash_eq or 0.0) + (sti or 0.0)) if (cash_eq is not None or sti is not None) else None

    cur_assets    = _f(bal.get("totalCurrentAssets"))
    cur_liab      = _f(bal.get("totalCurrentLiabilities"))
    inventory     = _f(bal.get("inventory"))

    long_debt     = _f(bal.get("longTermDebt"))
    short_debt    = _f(bal.get("shortTermDebt"))
    sld_total     = _f(bal.get("shortLongTermDebtTotal"))
    totalDebt     = (long_debt or 0.0) + (short_debt or 0.0) if (long_debt is not None or short_debt is not None) else sld_total

    currentRatio  = (cur_assets / cur_liab) if isinstance(cur_assets, float) and isinstance(cur_liab, float) and cur_liab else pick("currentRatio")
    quickRatio    = ((cur_assets - inventory) / cur_liab) if all(isinstance(x, float) for x in [cur_assets, inventory, cur_liab]) and cur_liab else pick("quickRatio")
    debtToEquity  = (totalDebt / equity) if isinstance(totalDebt, float) and isinstance(equity, float) and equity else pick("debtToEquity")

    ocf   = _f(cfs.get("netCashProvidedByOperatingActivities") or cfs.get("operatingCashFlow") or cfs.get("operatingCashflow"))
    capex = _f(cfs.get("capitalExpenditure") or cfs.get("capitalExpenditures"))
    freeCashflow = (ocf - capex) if isinstance(ocf, float) and isinstance(capex, float) else None

    shares_out = _f(
        (prof.get("sharesOutstanding") if prof else None) or
        (_latest_row(_read_json(FMP_DIR / f"{symbol}_KeyMetrics.json") or []).get("sharesOutstandingTTM") if (FMP_DIR / f"{symbol}_KeyMetrics.json").exists() else None) or
        (_latest_row(_read_json(FMP_DIR / f"{symbol}_KeyMetrics.json") or []).get("sharesOutstanding") if (FMP_DIR / f"{symbol}_KeyMetrics.json").exists() else None) or
        (inc.get("weightedAverageShsOutDil") if inc else None) or
        (inc.get("weightedAverageShsOut") if inc else None)
    )

    # TTM/Key Kennzahlen
    peRatio       = pick("peRatio")
    priceToBook   = pick("priceToBook")
    dividendYield = pick("dividendYield")
    payoutRatio   = pick("payoutRatio")
    pegRatio      = pick("pegRatio")

    bookValuePerShare    = pick("bookValuePerShare")
    cashPerShare         = pick("cashPerShare")
    freeCashFlowPerShare = pick("freeCashFlowPerShare")
    if freeCashFlowPerShare is None and isinstance(freeCashflow, float) and isinstance(shares_out, float) and shares_out:
        freeCashFlowPerShare = freeCashflow / shares_out

    cashToDebt  = (totalCash / totalDebt) if isinstance(totalCash, float) and isinstance(totalDebt, float) and totalDebt else None
    equityRatio = (equity / totalAssets) if isinstance(equity, float) and isinstance(totalAssets, float) and totalAssets else None
    fcfMargin   = (freeCashflow / revenue) if isinstance(freeCashflow, float) and isinstance(revenue, float) and revenue else None

    trailingPE     = peRatio
    freeCashFlow   = freeCashflow
    debtToAssets   = (totalDebt / totalAssets) if isinstance(totalDebt, float) and isinstance(totalAssets, float) and totalAssets else None

    # Growth
    earningsGrowth = None
    if len(inc_list) >= 2:
        n0 = _f(inc_list[0].get("netIncome"))
        n1 = _f(inc_list[1].get("netIncome"))
        if isinstance(n0, float) and isinstance(n1, float) and n1:
            earningsGrowth = (n0 - n1) / abs(n1)
        else:
            e0 = _f(inc_list[0].get("eps"))
            e1 = _f(inc_list[1].get("eps"))
            if isinstance(e0, float) and isinstance(e1, float) and e1:
                earningsGrowth = (e0 - e1) / abs(e1)

    epsGrowth = None
    if len(inc_list) >= 2:
        e0 = _f(inc_list[0].get("eps"))
        e1 = _f(inc_list[1].get("eps"))
        if isinstance(e0, float) and isinstance(e1, float) and e1:
            epsGrowth = (e0 - e1) / abs(e1)
        elif isinstance(shares_out, float) and shares_out:
            ni0 = _f(inc_list[0].get("netIncome"))
            ni1 = _f(inc_list[1].get("netIncome"))
            if isinstance(ni0, float) and isinstance(ni1, float) and ni1:
                eps0 = ni0 / shares_out
                eps1 = ni1 / shares_out
                if eps1:
                    epsGrowth = (eps0 - eps1) / abs(eps1)

    sgaTrend = None
    if len(inc_list) >= 3:
        def _ratio(i):
            rev = _f(inc_list[i].get("revenue") or inc_list[i].get("totalRevenue"))
            sga = _f(inc_list[i].get("sellingGeneralAdministrative"))
            if isinstance(rev, float) and rev and isinstance(sga, float):
                return sga / rev
            return None
        r0, r1, r2 = _ratio(0), _ratio(1), _ratio(2)
        if all(isinstance(x, float) for x in (r0, r1, r2)):
            sgaTrend = (r0 < r1) and (r1 < r2)

    if pegRatio is None and isinstance(trailingPE, float) and isinstance(earningsGrowth, float) and earningsGrowth > 0:
        pegRatio = trailingPE / (earningsGrowth * 100.0)

    out = {k: v for k, v in {
        "marketCap": marketCap,
        "peRatio": peRatio,
        "priceToBook": priceToBook,
        "dividendYield": dividendYield,
        "payoutRatio": payoutRatio,
        "sector": sector,
        "industry": industry,
        "beta": beta,
        "pegRatio": pegRatio,
        "revenue": revenue,
        "profitMargin": profitMargin,
        "revenueGrowth": revenueGrowth,
        "totalAssets": totalAssets,
        "totalDebt": totalDebt,
        "totalCash": totalCash,
        "totalStockholdersEquity": equity,
        "sharesOutstanding": shares_out,
        "currentRatio": currentRatio,
        "quickRatio": quickRatio,
        "debtToEquity": debtToEquity,
        "freeCashflow": freeCashflow,
        "cashPerShare": cashPerShare,
        "freeCashFlowPerShare": freeCashFlowPerShare,
        "cashToDebt": cashToDebt,
        "equityRatio": equityRatio,
        "fcfMargin": fcfMargin,
        "bookValuePerShare": bookValuePerShare,
        "eps": eps,
        # Aliasse
        "trailingPE": trailingPE,
        "freeCashFlow": freeCashflow,
        "debtToAssets": debtToAssets,
        "earningsGrowth": earningsGrowth,
        "epsGrowth": epsGrowth,
        "sgaTrend": sgaTrend,
    }.items() if v is not None}
    return out

def build_doc(symbol: str, metrics: Dict[str, Any], missing_fields: List[str]) -> Dict[str, Any]:
    today = str(datetime.now(UTC).date())
    return {
        "_index": ES_INDEX,
        "_id": f"{symbol}|{today}|fmp",
        "_source": {
            "symbol": symbol,
            "date": today,
            "source": "fmp",
            "ingested_at": datetime.now(UTC).isoformat(),
            "missing_fields": missing_fields,
            **metrics
        },
    }

# ===================== Historischer Backfill (vereinheitlicht als "fmp") =====================
def _load_all(symbol: str) -> Dict[str, Any]:
    files = {
        "Profile":           FMP_DIR / f"{symbol}_Profile.json",
        "IncomeStatement":   FMP_DIR / f"{symbol}_IncomeStatement.json",
        "BalanceSheet":      FMP_DIR / f"{symbol}_BalanceSheet.json",
        "CashflowStatement": FMP_DIR / f"{symbol}_CashflowStatement.json",
        "KeyMetrics":        FMP_DIR / f"{symbol}_KeyMetrics.json",
        "Ratios":            FMP_DIR / f"{symbol}_Ratios.json",
    }
    data = {k: (_read_json(p) if p.exists() else None) for k, p in files.items()}
    prof = data.get("Profile")
    if isinstance(prof, list) and prof: prof = prof[0]
    if not isinstance(prof, dict): prof = {}
    data["Profile"] = prof
    return data

def build_missing_fields(doc: dict) -> List[str]:
    req = ["peRatio","revenueGrowth","earningsGrowth","dividendYield","payoutRatio","marketCap"]
    return [k for k in req if doc.get(k) is None]
def _compute_derived_turnaround_metrics(doc: dict) -> dict:
    """
    Berechnet für historische Dokumente Turnaround-Kennzahlen nach,
    ähnlich wie in build_metrics_fmp:
    - cashToDebt
    - equityRatio
    - fcfMargin
    """
    # Basisgrößen
    revenue = _f(doc.get("revenue") or doc.get("totalRevenue"))

    totalAssets = _f(doc.get("totalAssets"))
    equity = _f(doc.get("totalStockholdersEquity") or doc.get("totalStockholderEquity"))

    cash_eq = _f(doc.get("cashAndCashEquivalents"))
    sti = _f(doc.get("shortTermInvestments"))
    totalCash = ((cash_eq or 0.0) + (sti or 0.0)) if (cash_eq is not None or sti is not None) else None

    long_debt = _f(doc.get("longTermDebt"))
    short_debt = _f(doc.get("shortTermDebt"))
    sld_total = _f(doc.get("shortLongTermDebtTotal"))
    totalDebt = (long_debt or 0.0) + (short_debt or 0.0) if (long_debt is not None or short_debt is not None) else sld_total

    if doc.get("totalDebt") is None and totalDebt is not None:
        doc["totalDebt"] = totalDebt

    ocf = _f(
        doc.get("netCashProvidedByOperatingActivities")
        or doc.get("operatingCashFlow")
        or doc.get("operatingCashflow")
    )
    capex = _f(doc.get("capitalExpenditure") or doc.get("capitalExpenditures"))
    freeCashflow = (ocf - capex) if isinstance(ocf, float) and isinstance(capex, float) else None

    if doc.get("freeCashflow") is None and freeCashflow is not None:
        doc["freeCashflow"] = freeCashflow
        doc["freeCashFlow"] = freeCashflow  # Alias wie im Heute-Dokument

    # cashToDebt
    if doc.get("cashToDebt") is None and isinstance(totalCash, float) and isinstance(totalDebt, float) and totalDebt:
        doc["cashToDebt"] = totalCash / totalDebt

    # equityRatio
    if doc.get("equityRatio") is None and isinstance(equity, float) and isinstance(totalAssets, float) and totalAssets:
        doc["equityRatio"] = equity / totalAssets

    # fcfMargin
    if doc.get("fcfMargin") is None and isinstance(freeCashflow, float) and isinstance(revenue, float) and revenue:
        doc["fcfMargin"] = freeCashflow / revenue

    return doc

def _enrich_historical_metrics(doc: dict) -> dict:
    """
    Versucht, historische Dokus so aufzubereiten wie build_metrics_fmp:
    - normalisierte Felder (peRatio, priceToBook, dividendYield, payoutRatio, pegRatio, ...)
    - PEG-Berechnung wie im Heute-Dokument, falls möglich.
    """
    # --- 1) peRatio & Co aus Roh-Feldern mappen ---
    # Wenn peRatio noch nicht gesetzt ist, aus FMP-Feldern holen
    if doc.get("peRatio") is None:
        for k in ("priceEarningsRatioTTM", "priceEarningsRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["peRatio"] = v
                break

    if doc.get("priceToBook") is None:
        for k in ("priceToBookRatio", "pbRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["priceToBook"] = v
                break

    if doc.get("dividendYield") is None:
        for k in ("dividendYieldTTM", "dividendYield"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["dividendYield"] = v
                break

    if doc.get("payoutRatio") is None:
        for k in ("payoutRatioTTM", "payoutRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["payoutRatio"] = v
                break

    if doc.get("debtToEquity") is None:
        for k in ("debtToEquityTTM", "debtEquityRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["debtToEquity"] = v
                break

    if doc.get("currentRatio") is None:
        for k in ("currentRatioTTM", "currentRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["currentRatio"] = v
                break

    if doc.get("quickRatio") is None:
        for k in ("quickRatioTTM", "quickRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["quickRatio"] = v
                break

    # --- 2) PEG aus FMP-Feldern mappen ---
    if doc.get("pegRatio") is None:
        for k in ("pegRatioTTM", "pegRatio",
                  "priceEarningsToGrowthRatioTTM", "priceEarningsToGrowthRatio"):
            v = doc.get(k)
            if isinstance(v, (int, float)):
                doc["pegRatio"] = v
                break

    # --- 3) Falls immer noch None: PEG selbst berechnen wie im Heute-Dokument ---
    pe = doc.get("peRatio") or doc.get("trailingPE")
    earnings_growth = doc.get("earningsGrowth")
    if doc.get("pegRatio") is None and isinstance(pe, (int, float)) and isinstance(earnings_growth, (int, float)) and earnings_growth > 0:
        # gleiches Schema wie in build_metrics_fmp
        doc["pegRatio"] = pe / (earnings_growth * 100.0)

    return doc

def build_historical_actions(symbol: str) -> List[Dict[str, Any]]:
    d = _load_all(symbol)
    prof = d.get("Profile") or {}

    inc_list = d.get("IncomeStatement") if isinstance(d.get("IncomeStatement"), list) else []
    bal_list = d.get("BalanceSheet") if isinstance(d.get("BalanceSheet"), list) else []
    cfs_list = d.get("CashflowStatement") if isinstance(d.get("CashflowStatement"), list) else []
    kms_list = d.get("KeyMetrics") if isinstance(d.get("KeyMetrics"), list) else []
    rat_list = d.get("Ratios") if isinstance(d.get("Ratios"), list) else []

    # alle verfügbaren Datumsstempel einsammeln
    dates = set()
    for seq in (inc_list, bal_list, cfs_list, kms_list, rat_list):
        for dd, _ in _rows_with_date(seq):
            dates.add(dd)
    if not dates:
        return []

    actions: List[Dict[str, Any]] = []
    today_iso = datetime.now(UTC).date().isoformat()

    for dd in sorted(dates):
        dd_iso = dd.isoformat()

        # GUARD: Wenn ein historischer Datensatz dasselbe Datum wie „heute“ hat,
        # würde _op_type=create mit der heutigen ID kollidieren → überspringen.
        if dd_iso == today_iso:
            continue

        doc = {
            "symbol": symbol,
            "date": dd_iso,
            "source": "fmp",  # vereinheitlicht
            "ingested_at": datetime.now(UTC).isoformat()
        }

        # pro Quelle nur die Einträge mit exakt diesem Datum einmischen
        for seq in (kms_list, rat_list, inc_list, bal_list, cfs_list):
            for ddd, row in _rows_with_date(seq):
                if ddd == dd and isinstance(row, dict):
                    _merge_dict(doc, row, prefer_existing=True)

        # Profile (ohne Datum) einmalig
        if isinstance(prof, dict):
            _merge_dict(doc, prof, prefer_existing=True)

        # 1) numerische Roh-Felder in Floats umwandeln
        _normalize_numeric_fields(doc)

        # 2) Historische Kennzahlen wie im Heute-Dokument "anreichern"
        doc = _enrich_historical_metrics(doc)

        # 3) Turnaround-Kennzahlen (cashToDebt, equityRatio, fcfMargin) nachberechnen
        doc = _compute_derived_turnaround_metrics(doc)

        # 4) ggf. neu hinzugekommene Felder nochmal normalisieren
        _normalize_numeric_fields(doc)


        miss = build_missing_fields(doc)
        if miss:
            doc["missing_fields"] = miss


        actions.append({
            "_op_type": "create",                       # nichts überschreiben
            "_index": ES_INDEX,
            "_id": f"{symbol}|{dd_iso}|fmp",            # konsistenter ID-Suffix
            "_source": doc
        })

    return actions

# ===================== Lauf =====================
def run(batch_flush: int = 500):
    print(es_healthcheck(es))
    ensure_index(es, ES_INDEX)

    if not FMP_DIR.exists():
        raise FileNotFoundError(f"FMP-Datenordner nicht gefunden: {FMP_DIR}")

    symbols = _discover_symbols(FMP_DIR)
    random.shuffle(symbols)

    buffer, written = [], 0
    for i, sym in enumerate(symbols, 1):
        try:
            # 1) Datei-Vollständigkeit prüfen (STRICT steuert Skip)
            if not _has_all_required_files(sym):
                msg = f"⚠️ {sym}: unvollständige JSON-Dateien"
                if STRICT_MODE:
                    print(msg + " — wird übersprungen (STRICT).")
                    continue
                else:
                    print(msg + " — wird dennoch versucht (STRICT=0).")

            # 2) HEUTE-Dokument
            metrics = build_metrics_fmp(sym, FMP_DIR)
            if not metrics:
                print(f"⚠️ {sym}: keine Metriken extrahiert — wird übersprungen.")
                continue

            fehlend = _missing_required_fields(metrics)
            if STRICT_MODE and fehlend:
                print(f"⚠️ {sym}: wichtige Kennzahlen fehlen → {', '.join(fehlend)} — wird übersprungen (STRICT).")
            else:
                if fehlend:
                    print(f"ℹ️  {sym}: fehlende Kennzahlen (wird dennoch gespeichert) → {', '.join(fehlend)}")
                buffer.append(build_doc(sym, metrics, fehlend))

            # 3) HISTORIE: alle Jahre aus lokalen JSONs (eigene IDs, _op_type=create)
            hist_actions = build_historical_actions(sym)
            if hist_actions:
                buffer.extend(hist_actions)

            # 4) Bulk flushen
            if len(buffer) >= batch_flush:
                helpers.bulk(es, buffer, raise_on_error=False)
                written += len(buffer)
                buffer.clear()
                print(f"[{i}/{len(symbols)}] {written} Dokumente gespeichert...")

        except Exception as e:
            print(f"[FEHLER] {sym}: {e}")

    if buffer:
        helpers.bulk(es, buffer, raise_on_error=False)
        written += len(buffer)

    print(f"✅ FMP-Ingest fertig. Gesamt gespeichert: {written} Dokumente in '{ES_INDEX}'.")

if __name__ == "__main__":
    run()
