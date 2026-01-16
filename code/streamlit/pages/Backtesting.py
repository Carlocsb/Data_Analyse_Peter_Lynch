# pages/Top_10_Dynamisch.py
import os, sys, math, uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
import importlib
import numpy as np

# ==========================================================
# Pfad zur src-Ebene
# ==========================================================
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src import lynch_criteria
importlib.reload(lynch_criteria)
CATEGORIES = lynch_criteria.CATEGORIES

from src.funktionen import (
    get_es_connection,
    score_row,
    enrich_document_fields,
)

# ==========================================================
# Page setup
# ==========================================================
st.set_page_config(page_title="Top 10 – dynamisch", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png", caption="")
st.title("📊 Top 10 Aktien je Peter-Lynch-Kategorie (dynamisch)")
st.markdown("*(Daten aus Elasticsearch – bewertet nach Lynch-Kriterien)*")

# ==========================================================
# ES Setup
# ==========================================================
es = get_es_connection()
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "stocks")

# ==========================================================
# ✅ UI-State Cache (ES) mit TTL 60 min
# ==========================================================
STATE_INDEX = os.getenv("UI_STATE_INDEX", "ui_state_top10")
TTL_MINUTES = int(os.getenv("UI_STATE_TTL_MINUTES", "60"))

def ensure_state_index(es_client):
    try:
        if es_client.indices.exists(index=STATE_INDEX):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "client_id": {"type": "keyword"},
                    "saved_at": {"type": "date"},
                    "expires_at": {"type": "date"},
                    "payload": {"type": "object", "enabled": True},
                }
            }
        }
        es_client.indices.create(index=STATE_INDEX, body=mapping)
    except Exception:
        # wenn es z.B. schon existiert oder Rechte fehlen -> nicht hart abbrechen
        pass

def _now_utc():
    return datetime.now(timezone.utc)

def _parse_iso(dt_str: str):
    if not dt_str:
        return None
    try:
        # isoformat kann "Z" enthalten – sauber behandeln
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def load_cached_state(es_client, client_id: str):
    try:
        res = es_client.get(index=STATE_INDEX, id=client_id)
        doc = res.get("_source", {})
        exp = _parse_iso(doc.get("expires_at"))
        if not exp or exp < _now_utc():
            return None
        return doc.get("payload")
    except Exception:
        return None

def save_cached_state(es_client, client_id: str, payload: dict):
    try:
        now = _now_utc()
        body = {
            "client_id": client_id,
            "saved_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=TTL_MINUTES)).isoformat(),
            "payload": payload,
        }
        es_client.index(index=STATE_INDEX, id=client_id, document=body, refresh=True)
    except Exception:
        pass

ensure_state_index(es)

# Client-ID pro Browser-Session
if "client_id" not in st.session_state:
    st.session_state["client_id"] = str(uuid.uuid4())

# ==========================================================
# Sidebar Defaults aus Cache laden (einmalig pro Session)
# ==========================================================
DEFAULT_START = datetime(2009, 12, 1).date()
DEFAULT_END   = datetime(2010, 5, 20).date()

SOURCE_OPTIONS = ["Only FMP", "Only yfinance", "Only Alpha Vantage"]
QUARTER_OPTIONS = ["Q1", "Q2", "Q3", "Q4"]

CACHE_BOOT_KEY = "top10_cache_bootstrapped"

if not st.session_state.get(CACHE_BOOT_KEY, False):
    cached = load_cached_state(es, st.session_state["client_id"])
    if isinstance(cached, dict):
        # Filter
        filt = cached.get("filters", {})
        st.session_state.setdefault("source_choice", filt.get("source_choice", "Only FMP"))
        st.session_state.setdefault("quarter", filt.get("quarter", "Q1"))

        # dates: als iso gespeichert
        sd = filt.get("start_date")
        ed = filt.get("end_date")
        try:
            st.session_state.setdefault(
                "start_date",
                pd.to_datetime(sd).date() if sd else DEFAULT_START
            )
        except Exception:
            st.session_state.setdefault("start_date", DEFAULT_START)

        try:
            st.session_state.setdefault(
                "end_date",
                pd.to_datetime(ed).date() if ed else DEFAULT_END
            )
        except Exception:
            st.session_state.setdefault("end_date", DEFAULT_END)

        # Kategorie
        st.session_state.setdefault("kategorie", cached.get("kategorie"))

        # Top10 export (optional)
        if cached.get("top10_dynamisch_export"):
            st.session_state["top10_dynamisch_export"] = cached["top10_dynamisch_export"]
            st.sidebar.info("♻️ Top10-Export aus Cache geladen (TTL).")
    st.session_state[CACHE_BOOT_KEY] = True

# ==========================================================
# Sidebar Controls (mit Keys, damit nichts „weg“ ist)
# ==========================================================
st.sidebar.header("⚙️ Einstellungen")

# robust index bestimmen
def _idx(options, val, fallback=0):
    try:
        return options.index(val)
    except Exception:
        return fallback

source_choice = st.sidebar.selectbox(
    "Quelle",
    SOURCE_OPTIONS,
    index=_idx(SOURCE_OPTIONS, st.session_state.get("source_choice", "Only FMP"), 0),
    key="source_choice",
)
quarter = st.sidebar.selectbox(
    "Quartal",
    QUARTER_OPTIONS,
    index=_idx(QUARTER_OPTIONS, st.session_state.get("quarter", "Q1"), 0),
    key="quarter",
)

MIN_DATE = datetime(2000, 1, 1).date()
MAX_DATE = datetime(2027, 12, 31).date()

start_date = st.sidebar.date_input(
    "Startdatum",
    value=st.session_state.get("start_date", DEFAULT_START),
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    key="start_date",
)

end_date = st.sidebar.date_input(
    "Enddatum",
    value=st.session_state.get("end_date", DEFAULT_END),
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    key="end_date",
)


if start_date > end_date:
    st.sidebar.error("Startdatum darf nicht nach Enddatum liegen.")
    st.stop()

START_DATE_STR = start_date.isoformat()
END_DATE_STR   = end_date.isoformat()
FIXED_QUARTER  = quarter
SOURCE_MODE    = source_choice

st.sidebar.caption(f"Zeitraum: {START_DATE_STR} .. {END_DATE_STR}")

# ==========================================================
# Source helper
# ==========================================================
def _source_term(mode: str):
    if mode == "Only FMP":
        return ["fmp"]
    if mode == "Only yfinance":
        return ["yfinance"]
    if mode == "Only Alpha Vantage":
        return ["alphavantage"]
    return ["fmp"]

SOURCES = _source_term(SOURCE_MODE)

# ==========================================================
# Load docs in window (source + date range + quarter)
# ==========================================================
must = [
    {
        "bool": {
            "should": (
                [{"term": {"source": s}} for s in SOURCES]
                + [{"term": {"source.keyword": s}} for s in SOURCES]
            ),
            "minimum_should_match": 1,
        }
    },
    {"range": {"date": {"gte": START_DATE_STR, "lte": END_DATE_STR}}},
    {
        "bool": {
            "should": [
                {"term": {"period": FIXED_QUARTER}},
                {"term": {"period.keyword": FIXED_QUARTER}},
            ],
            "minimum_should_match": 1,
        }
    },
]

query = {
    "size": 10000,
    "query": {"bool": {"must": must}},
    "sort": [{"date": {"order": "desc"}}, {"ingested_at": {"order": "desc"}}],
    "_source": True,
}

resp = es.search(index=ES_INDEX, body=query)
hits = [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
df = pd.DataFrame(hits)

if df.empty:
    st.warning("⚠️ Keine Daten nach Filter (Quelle/Zeitraum/Quartal).")
    # trotzdem Filter in Cache speichern, damit Sidebar beim nächsten Öffnen bleibt
    save_cached_state(es, st.session_state["client_id"], {
        "filters": {
            "source_choice": SOURCE_MODE,
            "quarter": FIXED_QUARTER,
            "start_date": START_DATE_STR,
            "end_date": END_DATE_STR,
        },
        "kategorie": st.session_state.get("kategorie"),
        "top10_dynamisch_export": st.session_state.get("top10_dynamisch_export"),
    })
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

# pro Symbol: neuestes Dokument im Fenster
df = (
    df.dropna(subset=["symbol", "date"])
      .sort_values(["symbol", "date"], ascending=[True, False])
      .drop_duplicates("symbol", keep="first")
      .reset_index(drop=True)
)

# Enrichment (keine History-YoY!)
df = pd.DataFrame([
    enrich_document_fields(
        r.to_dict(),
        es=es,
        source_mode=SOURCE_MODE,
        fill_growth_from_history=False,
    )
    for _, r in df.iterrows()
])

# Optional: FCF bauen
def build_fcf_if_possible(d: dict) -> dict:
    out = dict(d)
    if out.get("freeCashFlow") is None:
        ocf = (
            out.get("operatingCashflow")
            or out.get("operatingCashFlow")
            or out.get("netCashProvidedByOperatingActivities")
        )
        capex = out.get("capitalExpenditures") or out.get("capitalExpenditure")
        if ocf is not None and capex is not None:
            try:
                out["freeCashFlow"] = float(ocf) - abs(float(capex))
            except Exception:
                pass

    if out.get("freeCashFlowPerShare") is None and out.get("freeCashFlow") is not None and out.get("sharesOutstanding"):
        try:
            out["freeCashFlowPerShare"] = float(out["freeCashFlow"]) / float(out["sharesOutstanding"])
        except Exception:
            pass
    return out

df = pd.DataFrame([build_fcf_if_possible(r.to_dict()) for _, r in df.iterrows()])

for col in ["marketCap", "peRatio"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "marketCap" in df.columns:
    df["MarketCap (Mrd USD)"] = (df["marketCap"] / 1e9).round(1)

# ==========================================================
# QoQ helpers
# ==========================================================
FIELD_FALLBACKS = {
    "revenue":   ["revenue", "totalRevenue", "revenueTTM"],
    "netIncome": ["netIncome", "netIncomeApplicableToCommonShares"],
    "eps":       ["eps", "epsDiluted", "epsdiluted", "trailingEps"],
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

def _as_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def compute_qoq_growth(curr_doc: dict, prev_doc: dict, field: str):
    if not isinstance(curr_doc, dict) or not isinstance(prev_doc, dict):
        return None
    v_now = _as_float(_get_with_fallback(curr_doc, field))
    v_prev = _as_float(_get_with_fallback(prev_doc, field))
    if v_now is None or v_prev in (None, 0):
        return None
    return (v_now / v_prev) - 1.0

def _prev_period_and_year(period: str, year: int):
    p = str(period).upper()
    mapping = {"Q1": ("Q4", year - 1), "Q2": ("Q1", year), "Q3": ("Q2", year), "Q4": ("Q3", year)}
    return mapping.get(p, (None, None))

def load_prev_quarter_doc(es_client, symbol: str, source_mode: str, curr_doc: dict):
    period = str(curr_doc.get("period") or "").upper()
    year = curr_doc.get("calendarYear")

    if year is None and curr_doc.get("date") is not None:
        try:
            year = pd.to_datetime(curr_doc["date"]).year
        except Exception:
            year = None

    if not period or year is None:
        return None

    try:
        year_int = int(str(year))
    except Exception:
        return None

    prev_p, prev_y = _prev_period_and_year(period, year_int)
    if not prev_p:
        return None

    must_prev = [
        {"bool": {"should": [{"term": {"symbol": symbol}}, {"term": {"symbol.keyword": symbol}}], "minimum_should_match": 1}},
        {
            "bool": {
                "should": (
                    [{"term": {"source": s}} for s in _source_term(source_mode)]
                    + [{"term": {"source.keyword": s}} for s in _source_term(source_mode)]
                ),
                "minimum_should_match": 1,
            }
        },
        {"bool": {"should": [{"term": {"period": prev_p}}, {"term": {"period.keyword": prev_p}}], "minimum_should_match": 1}},
        {
            "bool": {
                "should": [
                    {"term": {"calendarYear": prev_y}},
                    {"term": {"calendarYear.keyword": str(prev_y)}},
                    {"term": {"calendarYear": str(prev_y)}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]

    body = {
        "size": 1,
        "sort": [{"date": {"order": "desc"}}, {"ingested_at": {"order": "desc"}}],
        "_source": True,
        "query": {"bool": {"must": must_prev}},
    }
    res = es_client.search(index=ES_INDEX, body=body)
    hits_prev = res.get("hits", {}).get("hits", [])
    return hits_prev[0]["_source"] if hits_prev else None

def apply_qoq_growth_fields(row_dict: dict) -> dict:
    out = dict(row_dict)
    sym = out.get("symbol")
    if not sym:
        return out

    prev_doc = load_prev_quarter_doc(es, sym, SOURCE_MODE, out)
    if not isinstance(prev_doc, dict):
        return out

    rg = compute_qoq_growth(out, prev_doc, "revenue")
    eg = compute_qoq_growth(out, prev_doc, "netIncome")
    if eg is None:
        eg = compute_qoq_growth(out, prev_doc, "eps")

    if rg is not None:
        out["revenueGrowth"] = rg
    if eg is not None:
        out["earningsGrowth"] = eg
        out["epsGrowth"] = eg

    return out

df = pd.DataFrame([apply_qoq_growth_fields(r.to_dict()) for _, r in df.iterrows()])

# ==========================================================
# Kriterien + Labels
# ==========================================================
def make_criteria_with_labels():
    FIELD_ALIAS = {"trailingPE": "peRatio"}
    LABEL_MAP = {
        "earningsGrowth": "Gewinnwachstum",
        "epsGrowth": "EPS-Wachstum",
        "eps": "EPS",
        "dividendYield": "Dividendenrendite",
        "payoutRatio": "Payout Ratio",
        "revenueGrowth": "Umsatzwachstum",
        "peRatio": "KGV",
        "priceToBook": "P/B",
        "marketCap": "Marktkapitalisierung",
        "freeCashFlow": "Free Cash Flow",
        "freeCashFlowPerShare": "FCF/Aktie",
        "debtToAssets": "Debt/Assets",
        "cashPerShare": "Cash/Aktie",
        "bookValuePerShare": "Buchwert/Aktie",
        "totalDebt": "Gesamtschulden",
        "sector": "Sektor",
        "currentRatio": "Current Ratio",
        "quickRatio": "Quick Ratio",
        "fcfMargin": "FCF-Marge",
        "profitMargin": "Profit-Marge",
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

CRITERIA = make_criteria_with_labels()

# Kategorie-Widget (persistiert)
kategorie_default = st.session_state.get("kategorie")
if kategorie_default not in list(CRITERIA.keys()):
    kategorie_default = list(CRITERIA.keys())[0]

kategorie = st.selectbox(
    "Kategorie wählen:",
    list(CRITERIA.keys()),
    index=list(CRITERIA.keys()).index(kategorie_default),
    key="kategorie",
)
criteria = CRITERIA[kategorie]

# ==========================================================
# Bewertung / Ranking
# ==========================================================
def evaluate_stock(row, criteria_local):
    score = score_row(row, criteria_local)
    max_score = len(criteria_local)

    results = []
    getv = row.get if isinstance(row, dict) else row.__getitem__

    for item in criteria_local:
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

        if isinstance(val, (int, float)) and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            if field in {"peRatio", "earningsGrowth", "epsGrowth"} and val <= 0:
                hinweis = "Bewertung nicht möglich: Wert ist 0 oder negativ."
                ok = False
            else:
                try:
                    ok = bool(rule(val))
                except Exception:
                    ok = False

        results.append({
            "Kennzahl": label,
            "Feld": field,
            "Istwert": val,
            "Erfüllt": ok,
            "Optional": optional,
            "Hinweis": hinweis
        })

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

st.caption(f"Quelle: {SOURCE_MODE} | Zeitraum: {START_DATE_STR}..{END_DATE_STR} | Quartal: {FIXED_QUARTER}")
st.markdown(f"### 📈 Ranking – {kategorie}")

cols_to_show = ["symbol", "Score", "MaxScore", "Score %"]
if "MarketCap (Mrd USD)" in df.columns:
    cols_to_show.insert(1, "MarketCap (Mrd USD)")

st.dataframe(
    df[cols_to_show].head(10).style.format({"MarketCap (Mrd USD)": "{:,.1f}", "Score %": "{:.0f} %"}),
    use_container_width=True,
)

# ==========================================================
# Detailansicht
# ==========================================================
aktie = st.selectbox("Wähle eine Aktie für Details:", df["symbol"].head(10))
row = df[df["symbol"] == aktie].iloc[0]
curr_doc = row.to_dict()

st.markdown("---")
st.subheader(f"🔍 Detailansicht: {aktie}")

def _fmt_value(x, field_name):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    if any(s in field_name.lower() for s in ["growth", "margin", "yield"]):
        try:
            return f"{float(x)*100:.2f} %"
        except Exception:
            return x
    try:
        return f"{float(x):.4f}" if isinstance(x, (int, float)) else x
    except Exception:
        return x

def _icon(ok, optional=False):
    base = "✅" if ok else "❌"
    return base + (" *(optional)*" if optional else "")

st.markdown("### Aktuell (aus Zeitraum/Quartal – exakt für Ranking genutzt)")

for item in row["Details"]:
    label = item["Kennzahl"]
    field = item["Feld"]
    hinweis = item.get("Hinweis")
    val = curr_doc.get(field)

    if hinweis:
        st.write(f"{_icon(False, item['Optional'])} **{label}** → _{hinweis}_")
    else:
        st.write(f"{_icon(item['Erfüllt'], item['Optional'])} **{label}** → **Ist:** {_fmt_value(val, field)}")

# ==========================================================
# Rohdaten (Audit Trail im Fenster)
# ==========================================================
def load_raw_docs_for_symbol(es_client, symbol: str, start_date: str, end_date: str, source_mode: str):
    must_local = [
        {"bool": {"should": [{"term": {"symbol": symbol}}, {"term": {"symbol.keyword": symbol}}], "minimum_should_match": 1}},
        {
            "bool": {
                "should": (
                    [{"term": {"source": s}} for s in _source_term(source_mode)]
                    + [{"term": {"source.keyword": s}} for s in _source_term(source_mode)]
                ),
                "minimum_should_match": 1,
            }
        },
        {"range": {"date": {"gte": start_date, "lte": end_date}}},
    ]
    body = {
        "size": 500,
        "sort": [{"date": {"order": "desc"}}, {"ingested_at": {"order": "desc"}}],
        "_source": True,
        "query": {"bool": {"must": must_local}},
    }
    res = es_client.search(index=ES_INDEX, body=body)
    hits_local = [h["_source"] for h in res.get("hits", {}).get("hits", [])]
    return pd.DataFrame(hits_local)

st.markdown("---")
with st.expander("📄 Rohdaten anzeigen (Audit Trail im Fenster)", expanded=False):
    st.markdown("### 🔹 Verwendetes Dokument (angereichert – für Ranking/Score genutzt)")
    st.dataframe(pd.DataFrame(curr_doc.items(), columns=["Feld", "Wert"]), use_container_width=True)

    st.markdown("### 🔹 Alle Rohdokumente im Zeitraum (Fenster)")
    df_raw = load_raw_docs_for_symbol(es, aktie, START_DATE_STR, END_DATE_STR, SOURCE_MODE)

    if "period" in df_raw.columns:
        df_raw["_period"] = df_raw["period"].astype(str).str.upper().str.strip()
        df_raw = df_raw[df_raw["_period"] == FIXED_QUARTER].drop(columns=["_period"], errors="ignore")

    if df_raw.empty:
        st.info("Keine Rohdaten im Zeitraum gefunden.")
    else:
        if "date" in df_raw.columns:
            df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce", utc=True)

        cols_first = [c for c in ["symbol", "date", "period", "calendarYear", "source", "ingested_at"] if c in df_raw.columns]
        other_cols = [c for c in df_raw.columns if c not in cols_first]
        df_raw = df_raw[cols_first + other_cols]

        st.dataframe(df_raw.sort_values("date", ascending=False), use_container_width=True)

# ==========================================================
# ✅ Export Top10 (alle Kategorien) in Session State
# ==========================================================
top10_export = {}

for cat in CRITERIA.keys():
    criteria_cat = CRITERIA[cat]
    tmp = df.copy()

    scores_cat = tmp.apply(lambda r: evaluate_stock(r, criteria_cat), axis=1)
    tmp["Score"] = [s[0] for s in scores_cat]
    tmp["MaxScore"] = [s[1] for s in scores_cat]
    tmp["Score %"] = (tmp["Score"] / tmp["MaxScore"] * 100).round(1)

    sort_cols = ["Score %"]
    ascending = [False]
    if "marketCap" in tmp.columns:
        sort_cols.append("marketCap")
        ascending.append(False)

    tmp = tmp.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    top10_export[cat] = tmp["symbol"].dropna().astype(str).head(10).tolist()

top10_payload = {
    "meta": {
        "source": SOURCE_MODE,
        "quarter": FIXED_QUARTER,
        "start": START_DATE_STR,
        "end": END_DATE_STR,
        "ts": datetime.utcnow().isoformat(),
    },
    "top10_by_category": top10_export,
}

st.session_state["top10_dynamisch_export"] = top10_payload

# ==========================================================
# ✅ Cache in ES aktualisieren (Filters + Kategorie + Export)
# ==========================================================
cache_payload = {
    "filters": {
        "source_choice": SOURCE_MODE,
        "quarter": FIXED_QUARTER,
        "start_date": START_DATE_STR,
        "end_date": END_DATE_STR,
    },
    "kategorie": kategorie,
    "top10_dynamisch_export": top10_payload,
}
save_cached_state(es, st.session_state["client_id"], cache_payload)

st.sidebar.success(f"✅ Top10-Export gesetzt & gecached (TTL {TTL_MINUTES} min). Öffne jetzt die Portfolio-Page.") 