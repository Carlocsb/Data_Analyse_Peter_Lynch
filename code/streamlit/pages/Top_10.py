# pages/Top_10_Kategorien.py
import os, sys, math
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import importlib

# Pfad zur src-Ebene
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Modul laden & sicher neu laden
from src import lynch_criteria
importlib.reload(lynch_criteria)
CATEGORIES = lynch_criteria.CATEGORIES

from src.funktionen import (
    get_es_connection,
    load_data_from_es,
    render_source_selector,
    score_row,
)

# === 1️⃣ Setup ===
st.set_page_config(page_title="Top 10 Aktien je Peter-Lynch-Kategorie", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png", caption="")
st.title("📊 Top 10 Aktien je Peter-Lynch-Kategorie")
st.markdown("*(Daten live aus Elasticsearch – bewertet nach Lynch-Kriterien)*")

# === 2️⃣ Verbindung & Daten ===
es = get_es_connection()
source_mode = render_source_selector()
df = load_data_from_es(es, source_mode=source_mode)
# === Zeitfilter ===
# === Zeitfilter ===
st.sidebar.header("📅 Zeitraum / Quartal")

# 1) date sicher als datetime (für Fallback)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

# 2) calendarYear robust normalisieren (NaN/Inf -> NA)
#    Fallback: wenn calendarYear fehlt/NA ist, aus date.year ziehen
df["_calyear"] = pd.to_numeric(df.get("calendarYear"), errors="coerce")
df["_calyear"] = df["_calyear"].replace([float("inf"), float("-inf")], pd.NA)

if "date" in df.columns:
    df["_calyear"] = df["_calyear"].fillna(df["date"].dt.year)

# als "nullable int" speichern, damit NA möglich bleibt
df["_calyear"] = pd.to_numeric(df["_calyear"], errors="coerce").astype("Int64")

# 3) verfügbare Jahre bestimmen (ohne NA)
years = sorted(df["_calyear"].dropna().unique().astype(int).tolist())
if not years:
    st.warning("⚠️ Keine gültigen Jahreswerte gefunden (calendarYear/date).")
    st.stop()

selected_year = st.sidebar.selectbox("Jahr", years, index=len(years) - 1)

# 4) Quartal-Filter
quarter = st.sidebar.selectbox("Quartal", ["Alle", "Q1", "Q2", "Q3", "Q4"], index=0)

# 5) anwenden
df = df[df["_calyear"] == selected_year]

if quarter != "Alle":
    if "period" not in df.columns:
        st.warning("⚠️ Spalte 'period' fehlt – Quartal-Filter kann nicht angewendet werden.")
    else:
        df["_period"] = df["period"].astype(str).str.upper().str.strip()
        df = df[df["_period"] == quarter]


if df.empty:
    st.warning("⚠️ Keine Daten in Elasticsearch gefunden. Bitte Ingest laufen lassen.")
    st.stop()

for col in ["marketCap", "peRatio"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "stocks")

# === Source-Helfer ===
def _normalize_source_mode(s: str | None):
    if not s:
        return None
    key = s.strip().lower()
    if key.startswith("nur "):
        key = key[4:].strip()
    if key in ("alle", "all", "*"):
        return None
    if key in ("yf", "yfinance") or "yfinance" in key:
        return ["yfinance"]
    if key in ("av", "alphavantage", "alpha-vantage") or "alpha" in key:
        return ["alphavantage"]
    if "fmp" in key:
        return ["fmp"]
    return [key]

def _es_must_clauses(symbol: str, source_mode: str | None):
    must = [
        {
            "bool": {
                "should": [
                    {"term": {"symbol": symbol}},
                    {"match": {"symbol": symbol}},
                ],
                "minimum_should_match": 1,
            }
        }
    ]
    sources = _normalize_source_mode(source_mode)
    if isinstance(sources, list) and len(sources) == 1:
        must.append(
            {
                "bool": {
                    "should": [
                        {"term": {"source": sources[0]}},
                        {"match": {"source": sources[0]}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return must

# === Helfer ===
def _parse_es_date(s: str | None):
    if not isinstance(s, str):
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s2).astimezone(timezone.utc).date()
    except Exception:
        try:
            return datetime.fromisoformat(s.split("T")[0]).date()
        except Exception:
            return None

def _es_get_latest(symbol: str, source: str | None):
    body = {
        "size": 1,
        "sort": [{"date": {"order": "desc"}}],
        "_source": True,
        "query": {"bool": {"must": _es_must_clauses(symbol, source)}},
    }
    res = es.search(index=ES_INDEX, body=body)
    return res["hits"]["hits"][0]["_source"] if res["hits"]["hits"] else None

# === Jahr zurück ===
def es_get_prev_quarter_doc(symbol: str, source: str | None, latest_doc: dict):
    base_doc = None

    if isinstance(latest_doc, dict) and latest_doc.get("calendarYear") and latest_doc.get("period"):
        base_doc = latest_doc
    else:
        must_base = _es_must_clauses(symbol, source) + [
            {"exists": {"field": "calendarYear"}},
            {"exists": {"field": "period"}},
        ]

        body_base = {
            "size": 1,
            "sort": [{"date": {"order": "desc"}}],
            "_source": True,
            "query": {"bool": {"must": must_base}},
        }

        res_base = es.search(index=ES_INDEX, body=body_base)
        hits_base = res_base["hits"]["hits"]
        if not hits_base:
            return None, None
        base_doc = hits_base[0]["_source"]

    period = base_doc.get("period")
    year_now = base_doc.get("calendarYear")
    if not period or not year_now:
        return None, None

    try:
        year_now_int = int(str(year_now))
    except Exception:
        return None, None
    target_year = year_now_int - 1

    must_prev = _es_must_clauses(symbol, source) + [
        {
            "bool": {
                "should": [
                    {"term": {"calendarYear": target_year}},
                    {"term": {"calendarYear": str(target_year)}},
                    {"term": {"calendarYear.keyword": str(target_year)}},
                ],
                "minimum_should_match": 1,
            }
        },
        {
            "bool": {
                "should": [
                    {"term": {"period": period}},
                    {"term": {"period.keyword": period}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]

    body_prev = {
        "size": 1,
        "sort": [{"date": {"order": "desc"}}],
        "_source": True,
        "query": {"bool": {"must": must_prev}},
    }

    res_prev = es.search(index=ES_INDEX, body=body_prev)
    hits_prev = res_prev["hits"]["hits"]
    if not hits_prev:
        return None, None

    prev_doc = hits_prev[0]["_source"]

    d_now = _parse_es_date(base_doc.get("date"))
    d_prev = _parse_es_date(prev_doc.get("date"))
    dist = None
    if d_now and d_prev:
        dist = abs((d_now - d_prev).days)

    return prev_doc, dist

def has_year_back_data(
    symbol: str,
    source: str | None,
    *,
    latest_doc: dict,
    required_fields: list[str] | None = None,
):
    doc, dist = es_get_prev_quarter_doc(symbol, source, latest_doc)
    if not doc:
        return False, doc, dist
    if required_fields:
        for f in required_fields:
            if doc.get(f) is None:
                return False, doc, dist
    return True, doc, dist

# === Vorquartal-Suche ===
def es_get_prev_quarter_same_year(symbol: str, source: str | None, base_doc: dict):
    if not isinstance(base_doc, dict):
        return None, None

    period = str(base_doc.get("period", "")).upper()
    year = base_doc.get("calendarYear")
    if not period or year is None:
        return None, None

    try:
        year_int = int(str(year))
    except Exception:
        return None, None

    mapping = {
        "Q1": ("Q4", year_int - 1),
        "Q2": ("Q1", year_int),
        "Q3": ("Q2", year_int),
        "Q4": ("Q3", year_int),
    }
    prev_period, prev_year = mapping.get(period, (None, None))
    if not prev_period:
        return None, None

    must_prev = _es_must_clauses(symbol, source) + [
        {
            "bool": {
                "should": [
                    {"term": {"calendarYear": prev_year}},
                    {"term": {"calendarYear": str(prev_year)}},
                    {"term": {"calendarYear.keyword": str(prev_year)}},
                ],
                "minimum_should_match": 1,
            }
        },
        {
            "bool": {
                "should": [
                    {"term": {"period": prev_period}},
                    {"term": {"period.keyword": prev_period}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]

    body_prev = {
        "size": 1,
        "sort": [{"date": {"order": "desc"}}],
        "_source": True,
        "query": {"bool": {"must": must_prev}},
    }

    res_prev = es.search(index=ES_INDEX, body=body_prev)
    hits_prev = res_prev["hits"]["hits"]
    if not hits_prev:
        return None, None

    prev_doc = hits_prev[0]["_source"]

    d_now = _parse_es_date(base_doc.get("date"))
    d_prev = _parse_es_date(prev_doc.get("date"))
    dist = None
    if d_now and d_prev:
        dist = abs((d_now - d_prev).days)

    return prev_doc, dist

# === Fallbacks für Felder ===
FIELD_FALLBACKS = {
    "peRatio": ["trailingPE", "priceEarningsRatioTTM", "priceEarningsRatio"],
    "priceToBook": ["priceToBookRatio", "pbRatio"],
    "dividendYield": ["dividendYieldTTM", "dividendYield"],
    "payoutRatio": ["payoutRatioTTM", "payoutRatio"],
    "pegRatio": ["pegRatioTTM", "pegRatio"],
    "bookValuePerShare": ["bookValuePerShareTTM"],
    "cashPerShare": ["cashPerShareTTM"],
    "freeCashFlowPerShare": ["freeCashFlowPerShareTTM"],
    "debtToEquity": ["debtEquityRatio", "debtToEquityTTM"],
    "currentRatio": ["currentRatioTTM", "currentRatio"],
    "quickRatio": ["quickRatioTTM", "quickRatio"],
    "revenue": ["totalRevenue", "revenueTTM"],
    "eps": ["epsDiluted", "epsdiluted", "trailingEps"],
}

def _get_with_fallback(doc: dict, field: str):
    if not isinstance(doc, dict):
        return None
    if field in doc and doc[field] is not None:
        return doc[field]
    for alt in FIELD_FALLBACKS.get(field, []):
        if alt in doc and doc[alt] is not None:
            return doc[alt]
    return None

def compute_qoq_growth(curr_doc: dict, prev_doc: dict, field: str):
    if not isinstance(curr_doc, dict) or not isinstance(prev_doc, dict):
        return None
    v_now = _get_with_fallback(curr_doc, field)
    v_prev = _get_with_fallback(prev_doc, field)
    if not isinstance(v_now, (int, float)) or not isinstance(v_prev, (int, float)):
        return None
    if v_prev == 0:
        return None
    return (v_now / v_prev) - 1.0

# === Kriterien + Labels ===
def make_criteria_with_labels():
    FIELD_ALIAS = {
        "trailingPE": "peRatio",
    }
    LABEL_MAP = {
        "earningsGrowth":   "Gewinnwachstum",
        "epsGrowth":        "EPS-Wachstum",
        "eps":              "EPS",
        "dividendYield":    "Dividendenrendite",
        "payoutRatio":      "Payout Ratio",
        "revenueGrowth":    "Umsatzwachstum",
        "peRatio":          "KGV",
        "priceToBook":      "P/B",
        "marketCap":        "Marktkapitalisierung",
        "freeCashflow":     "Free Cash Flow",
        "freeCashFlow":     "Free Cash Flow",
        "freeCashFlowPerShare": "FCF/Aktie",
        "debtToAssets":     "Debt/Assets",
        "cashPerShare":     "Cash/Aktie",
        "bookValuePerShare":"Buchwert/Aktie",
        "totalDebt":        "Gesamtschulden",
        "sector":           "Sektor",
        "currentRatio":     "Current Ratio",
        "quickRatio":       "Quick Ratio",
        "fcfMargin":        "FCF-Marge",
        "profitMargin":     "Profit-Marge",
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

# === Kategorie-Auswahl ===
CRITERIA = make_criteria_with_labels()
kategorie = st.selectbox("Kategorie wählen:", list(CRITERIA.keys()))
criteria = CRITERIA[kategorie]

# === Filter ===
st.sidebar.header("🔍 Filter")

if "industry" in df.columns:
    industries = ["Alle"] + sorted(df["industry"].dropna().unique().tolist())
    selected_industry = st.sidebar.selectbox("Branche", industries)
    if selected_industry != "Alle":
        df = df[df["industry"] == selected_industry]

if "marketCap" in df.columns and not df["marketCap"].dropna().empty:
    df["marketCap_Mrd"] = df["marketCap"] / 1e9
    min_cap = float(df["marketCap_Mrd"].min())
    max_cap = float(df["marketCap_Mrd"].max())
    cap_range = st.sidebar.slider(
        "Marktkapitalisierung (in Mrd USD)",
        min_value=round(min_cap, 1),
        max_value=round(max_cap, 1),
        value=(round(min_cap, 1), round(max_cap, 1)),
        step=10.0 if (max_cap - min_cap) > 10 else 1.0,
    )
    df = df[(df["marketCap_Mrd"] >= cap_range[0]) & (df["marketCap_Mrd"] <= cap_range[1])]



# === Bewertung ===
def evaluate_stock(row, criteria):
    """
    Bewertet eine Zeile:
    Score = score_row(row)
    MaxScore = len(criteria) (immer 6 etc.)
    """
    score = score_row(row, criteria)
    max_score = len(criteria)

    results = []
    getv = row.get if isinstance(row, dict) else row.__getitem__

    for item in criteria:
        if len(item) == 2:
            field, rule = item
            label = field
            optional = False
        else:
            field, label, rule, optional = item

        try:
            val = getv(field)
        except Exception:
            val = None

        ok = False
        hinweis = None

        # gleiche Sonderlogik wie score_row: <=0 zählt als "nicht erfüllt"
        if isinstance(val, (int, float)):
            if field in {"peRatio", "earningsGrowth", "epsGrowth"} and val <= 0:
                hinweis = "Bewertung nicht möglich: Wert ist 0 oder negativ."
                ok = False
            else:
                try:
                    ok = bool(rule(val))
                except Exception:
                    ok = False

        results.append(
            {
                "Kennzahl": label,
                "Feld": field,
                "Istwert": val,
                "Erfüllt": ok,
                "Optional": optional,
                "Hinweis": hinweis,
            }
        )

    return score, max_score, results



scores = df.apply(lambda r: evaluate_stock(r, criteria), axis=1)
df["Score"] = [s[0] for s in scores]
df["MaxScore"] = [s[1] for s in scores]
df["Score %"] = (df["Score"] / df["MaxScore"] * 100).round(1)
df["Details"] = [s[2] for s in scores]

sort_cols = ["Score %"]
if "marketCap" in df.columns:
    sort_cols.append("marketCap")
df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

if "marketCap" in df.columns:
    df["MarketCap (Mrd USD)"] = (df["marketCap"] / 1e9).round(1)

# === Ranking ===
st.caption(f"Quelle: {source_mode}")
st.markdown(f"### 📈 Ranking – {kategorie}")
cols_to_show = ["symbol", "Score", "MaxScore", "Score %"]
if "MarketCap (Mrd USD)" in df.columns:
    cols_to_show.insert(1, "MarketCap (Mrd USD)")

st.dataframe(
    df[cols_to_show].head(10).style.format(
        {
            "MarketCap (Mrd USD)": "{:,.1f}",
            "Score %": "{:.0f} %",
        }
    ),
    use_container_width=True,
)

# === Detailansicht ===
aktie = st.selectbox("Wähle eine Aktie für Details:", df["symbol"].head(10))
row = df[df["symbol"] == aktie].iloc[0]

st.markdown("---")
st.subheader(f"🔍 Detailansicht: {aktie}")

preferred_source = None if (not source_mode) else source_mode
curr_doc = _es_get_latest(aktie, preferred_source)

ok_prev, prev_doc, prev_dist = has_year_back_data(
    aktie,
    preferred_source,
    latest_doc=curr_doc,
    required_fields=None,
)

# Vorquartale holen
prev_quarter_curr, _ = (None, None)
if isinstance(curr_doc, dict):
    prev_quarter_curr, _ = es_get_prev_quarter_same_year(aktie, preferred_source, curr_doc)

prev_quarter_prev, _ = (None, None)
if ok_prev and isinstance(prev_doc, dict):
    prev_quarter_prev, _ = es_get_prev_quarter_same_year(aktie, preferred_source, prev_doc)

# QoQ-Wachstum (EPS = Gewinn, Revenue = Umsatz)
umsatz_qoq_now = compute_qoq_growth(curr_doc, prev_quarter_curr, "revenue") if prev_quarter_curr else None
gewinn_qoq_now = compute_qoq_growth(curr_doc, prev_quarter_curr, "eps") if prev_quarter_curr else None

umsatz_qoq_prev = compute_qoq_growth(prev_doc, prev_quarter_prev, "revenue") if prev_quarter_prev else None
gewinn_qoq_prev = compute_qoq_growth(prev_doc, prev_quarter_prev, "eps") if prev_quarter_prev else None

# Mapping: für welche Felder ersetzen wir den Wert durch QoQ?
growth_fields_now = {
    "revenueGrowth": umsatz_qoq_now,
    "earningsGrowth": gewinn_qoq_now,
    "epsGrowth": gewinn_qoq_now,
}
growth_fields_prev = {
    "revenueGrowth": umsatz_qoq_prev,
    "earningsGrowth": gewinn_qoq_prev,
    "epsGrowth": gewinn_qoq_prev,
}

def _fmt_value(x, field_name):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    if any(s in field_name.lower() for s in ["growth", "margin", "yield"]):
        try:
            return f"{x*100:.2f} %"
        except Exception:
            return x
    return f"{x:.4f}" if isinstance(x, float) else x

def _icon(ok, optional=False):
    base = "✅" if ok else "❌"
    return base + (" *(optional)*" if optional else "")

col_l, col_r = st.columns(2)

# ---- Aktuell ----
with col_l:
    st.markdown("### Aktuell")
    for item in row["Details"]:
        label = item["Kennzahl"]
        field = item["Feld"]
        hinweis = item.get("Hinweis")

        override = growth_fields_now.get(field)
        if override is not None:
            now_v = override
        else:
            now_v = _get_with_fallback(curr_doc, field) if isinstance(curr_doc, dict) else row.get(field)

        if hinweis:
            # Nur Hinweis anzeigen – KEIN Wert
            st.write(
                f"{_icon(False, item['Optional'])} "
                f"**{label}** → _{hinweis}_"
            )
        else:
            # Normalfall: Wert anzeigen
            st.write(
                f"{_icon(item['Erfüllt'], item['Optional'])} "
                f"**{label}** → **Ist:** {_fmt_value(now_v, field)}"
            )



# ---- Vor ~1 Jahr ----
# QoQ-Werte sind schon berechnet:
# umsatz_qoq_prev, gewinn_qoq_prev

with col_r:
    st.markdown("### Vor ~1 Jahr")
    if not ok_prev or not isinstance(prev_doc, dict):
        ...
    else:
        # 🔧 NEU: Kopie des Dokuments für die Bewertung anlegen
        eval_doc = dict(prev_doc)
        if umsatz_qoq_prev is not None:
            eval_doc["revenueGrowth"] = umsatz_qoq_prev
        if gewinn_qoq_prev is not None:
            eval_doc["earningsGrowth"] = gewinn_qoq_prev
            eval_doc["epsGrowth"] = gewinn_qoq_prev

        # Bewertung jetzt auf Basis der QoQ-Werte
        prev_score, prev_maxscore, prev_details = evaluate_stock(eval_doc, criteria)
        prev_date = _parse_es_date(prev_doc.get("date"))

        base_doc_for_plain = prev_quarter_prev or prev_doc

        for item in prev_details:
            label = item["Kennzahl"]
            field = item["Feld"]
            ok_flag = item["Erfüllt"]
            opt_flag = item["Optional"]
            hinweis = item.get("Hinweis")

            override = growth_fields_prev.get(field)
            if override is not None:
                prev_v = override
            else:
                prev_v = _get_with_fallback(base_doc_for_plain, field)

            if hinweis:
                st.write(
                    f"{_icon(False, opt_flag)} "
                    f"**{label}** → _{hinweis}_"
                )
            else:
                st.write(
                    f"{_icon(ok_flag, opt_flag)} "
                    f"**{label}** → **Damals:** {_fmt_value(prev_v, field)}"
                )



