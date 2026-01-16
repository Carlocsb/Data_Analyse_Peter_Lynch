import json
from pathlib import Path

# === Pfade & Config ===
BASE_DIR = Path(__file__).resolve().parent              # .../code/API
PROJECTROOT = BASE_DIR.parents[1]                      # .../ (Projektwurzel)
FMP_DIR = PROJECTROOT / "data" / "sp_data" / "total_sp_data"  # FMP-Dateien
FILES = ["Profile", "IncomeStatement", "BalanceSheet", "CashflowStatement", "KeyMetrics", "Ratios"]

def _read_json(p: Path):
    """Liest JSON oder gibt [] bei leer/ungültig zurück"""
    if not p.exists():
        return []
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return []
        return json.loads(txt)
    except Exception:
        return []

def audit_fmp_folder(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"❌ FMP-Ordner nicht gefunden: {folder}")

    symbols = sorted({f.name.split("_")[0] for f in folder.glob("*_*.json")})
    print(f"🔍 Prüfe {len(symbols)} Symbole in: {folder}")

    report = []
    for sym in symbols:
        status = {}
        for name in FILES:
            p = folder / f"{sym}_{name}.json"
            js = _read_json(p)
            status[name] = "OK" if js else "EMPTY"

        empty_count = list(status.values()).count("EMPTY")
        completeness = "✅ Vollständig" if empty_count == 0 else "⚠️ Unvollständig"
        report.append((sym, completeness, empty_count, status))

        if empty_count >= 3:
            print(f"⚠️ {sym}: {empty_count} leere Dateien → {[k for k,v in status.items() if v == 'EMPTY']}")

    # Kurze Zusammenfassung
    total = len(report)
    full = sum(1 for _, c, _, _ in report if c.startswith("✅"))
    print(f"\n📊 Ergebnis: {full}/{total} vollständig ({(full/total)*100:.1f} %)")

    # Optional: CSV speichern
    out_path = BASE_DIR / "fmp_audit_report.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Symbol,Vollständig,Leere_Dateien,Details\n")
        for sym, completeness, empty_count, status in report:
            f.write(f"{sym},{completeness},{empty_count},{json.dumps(status)}\n")

    print(f"💾 Bericht gespeichert unter: {out_path}")

if __name__ == "__main__":
    audit_fmp_folder(FMP_DIR)
