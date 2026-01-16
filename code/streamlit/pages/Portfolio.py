import os
import sys
import uuid

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Pfad-Setup (Pages → src importierbar machen) ✅ vor src-imports
# ------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.lynch_criteria import CATEGORIES
from src.funktionen import (
    get_es_connection,
    load_data_from_es,
    load_industries,
    ensure_portfolio_index,
    build_portfolio_doc,
    save_portfolio,
    list_portfolios,
    load_portfolio,
    delete_portfolio,
    render_source_selector,
)

# -------- Session-Seed für stabile Widget-Keys --------
if "widget_seed" not in st.session_state:
    st.session_state["widget_seed"] = str(uuid.uuid4())

# -------- State defaults --------
if "current_portfolio_id" not in st.session_state:
    st.session_state["current_portfolio_id"] = None
if "portfolio_name" not in st.session_state:
    st.session_state["portfolio_name"] = ""

# === UI ===
st.set_page_config(page_title="Portfolio-Erstellung", layout="wide")
st.sidebar.image("assets/Logo-TH-Köln1.png")
st.title("📁 Portfolio-Zusammenstellung nach Peter Lynch")

# === ES + Source Mode (nur für Stocks/Industries-Load & Validierung) ===
es = get_es_connection()
source_mode = render_source_selector("📡 Datenquelle")
ensure_portfolio_index(es)

df_industries = load_industries(es, source_mode=source_mode)
df_stocks = load_data_from_es(es, source_mode=source_mode)

# ------------------------------------------------------------
# Hydration (geladenes Portfolio -> Defaults)
# Wichtig: wir setzen nur session_state für die Widget-Keys,
# und verwenden beim Widget KEIN default=..., um Streamlit-Warnungen zu vermeiden.
# ------------------------------------------------------------
if "hydrate_payload" in st.session_state:
    payload = st.session_state.pop("hydrate_payload")

    st.session_state["current_portfolio_id"] = payload.get("portfolio_id")
    st.session_state["portfolio_name"] = payload.get("portfolio_name", "")

    # Multiselect-Werte
    for cat, tickers in payload.get("auswahl", {}).items():
        st.session_state[f"ms_{cat}"] = list(dict.fromkeys(tickers))

    # Amounts
    for cat, symvals in payload.get("betraege", {}).items():
        for sym, amt in symvals.items():
            st.session_state[f"amt_{cat}_{sym}"] = float(amt)

# === Branchenfilter ===
st.sidebar.header("🔍 Filter")
industries = (
    ["Alle"] + sorted(df_industries["industry"].dropna().unique().tolist())
    if not df_industries.empty and "industry" in df_industries.columns
    else ["Alle"]
)
selected_industry = st.sidebar.selectbox("Branche", industries)

# ------------------------------------------------------------
# ✅ Top10 Export Status (von Top_10_Dynamisch)
# ------------------------------------------------------------
export = st.session_state.get("top10_dynamisch_export")
if not export:
    st.sidebar.warning("Kein Top10-Export gefunden. Bitte erst 'Top 10 – dynamisch' öffnen und Filter setzen.")
    top10_map = {}
else:
    top10_map = export.get("top10_by_category", {})
    meta = export.get("meta", {})
    st.sidebar.success(
        f"Top10 geladen: {meta.get('source','?')} | {meta.get('quarter','?')} | "
        f"{meta.get('start','?')}..{meta.get('end','?')}"
    )

# ------------------------------------------------------------
# Strategien nach Marktlage
# ------------------------------------------------------------
strategien = {
    "Markt fällt": {"Slow Grower": 30, "Stalwarts": 25, "Fast Grower": 10, "Cyclicals": 10, "Turn Around": 10, "Assets Player": 15},
    "Seitwärtsmarkt": {"Slow Grower": 20, "Stalwarts": 25, "Fast Grower": 20, "Cyclicals": 15, "Turn Around": 10, "Assets Player": 10},
    "Markt boomt": {"Slow Grower": 10, "Stalwarts": 15, "Fast Grower": 35, "Cyclicals": 20, "Turn Around": 10, "Assets Player": 10},
}

ALIAS = {
    "Slow Grower": "Slow Growers",
    "Stalwarts": "Stalwarts",
    "Fast Grower": "Fast Growers",
    "Cyclicals": "Cyclicals",
    "Turn Around": "Turnarounds",
    "Assets Player": "Asset Plays",
}

marktlage = st.selectbox("📉 Wähle die aktuelle Marktlage:", list(strategien.keys()))
verteilung = strategien[marktlage]
# ------------------------------------------------------------
# ✅ Auto-Fill: Auswahl + Beträge aus Top10-Export (einmalig)
# ------------------------------------------------------------
def build_auto_plan():
    # Regeln wie von dir gewünscht
    return {
        "Slow Grower": [("take", 2, [100, 100])],
        "Stalwarts":   [("take", 3, [100, 100, 50])],
        "Fast Grower": [("take", 2, [100, 100])],
        "Cyclicals":   [("take", 2, [100, 50])],
        "Turn Around": [("take", 1, [100])],
        "Assets Player":[("take", 1, [100])],
    }

def apply_autofill_from_top10(top10_map: dict, valid_symbols: set | None = None):
    plan = build_auto_plan()

    for ui_cat, rules in plan.items():
        export_cat = ALIAS.get(ui_cat, ui_cat)  # z.B. "Slow Growers"
        candidates = list(dict.fromkeys(top10_map.get(export_cat, [])))

        # optional: gegen ES-Universum validieren
        if valid_symbols:
            candidates = [s for s in candidates if s in valid_symbols]

        picked = []
        amounts = {}

        for rule in rules:
            _, n, amts = rule
            chosen = candidates[:n]
            picked.extend(chosen)
            for sym, amt in zip(chosen, amts):
                amounts[sym] = float(amt)

        picked = list(dict.fromkeys(picked))

        # ✅ Auswahl setzen (für multiselect)
        st.session_state[f"ms_{ui_cat}"] = picked

        # ✅ Beträge setzen (für number_input)
        for sym, amt in amounts.items():
            st.session_state[f"amt_{ui_cat}_{sym}"] = float(amt)

# --- nur ausführen, wenn Export existiert ---
export = st.session_state.get("top10_dynamisch_export")
top10_map = export.get("top10_by_category", {}) if export else {}
meta = export.get("meta", {}) if export else {}

# Validierungsset (Symbol-Universum)
valid_symbols = set(df_stocks["symbol"].dropna().unique()) if (not df_stocks.empty and "symbol" in df_stocks.columns) else None

# One-shot Gate: pro Export-Zeitraum + Quelle + Marktlage nur einmal auto-fill
autofill_sig = (
    str(meta.get("source", "")),
    str(meta.get("quarter", "")),
    str(meta.get("start", "")),
    str(meta.get("end", "")),
    str(marktlage),
)
autofill_key = "autofill_done_sig"

if export and st.session_state.get(autofill_key) != autofill_sig:
    apply_autofill_from_top10(top10_map, valid_symbols=valid_symbols)
    st.session_state[autofill_key] = autofill_sig

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📋 Empfohlene Verteilung:")
    for k, v in verteilung.items():
        st.write(f"- **{v}%** {k}")
with col2:
    st.subheader("📊 Visuelle Darstellung")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(verteilung.values(), labels=verteilung.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# ------------------------------------------------------------
# Auswahl (links) & Zusammenfassung (rechts)
# ------------------------------------------------------------
col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.subheader("🔎 Aktienauswahl je Kategorie")
    ausgewaehlte_aktien = {}

    # Validierung: Symbol-Universum aus df_stocks
    valid = set(df_stocks["symbol"].dropna().unique()) if not df_stocks.empty and "symbol" in df_stocks.columns else set()

    for k in verteilung:
        # Top10 aus Dynamisch-Export
        cat_key = ALIAS.get(k, k)  # z.B. "Slow Growers"
        options_base = list(dict.fromkeys(top10_map.get(cat_key, [])))

        # bereits vorhandene Auswahl (Hydration / Session)
        current_selection = list(dict.fromkeys(st.session_state.get(f"ms_{k}", [])))

        # ungültige raus (nur wenn valid verfügbar ist)
        if valid:
            current_selection = [t for t in current_selection if t in valid]
            options_base = [t for t in options_base if t in valid]

        # Union Top10 + bisherige Auswahl
        options = list(dict.fromkeys(options_base + current_selection))

        # WICHTIG: kein default=..., weil wir session_state ggf. bereits setzen
        selected = st.multiselect(
            f"{k} – Wähle Aktien aus:",
            options=options,
            key=f"ms_{k}",
        )

        ausgewaehlte_aktien[k] = list(dict.fromkeys(selected))

    # CSS für Expander
    st.markdown(
        """
        <style>
          div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            margin: 0 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 💵 Beträge sammeln
    betraege = {}
    with st.expander("💵 Betrag pro ausgewählter Aktie – klicken zum Ein-/Ausklappen", expanded=False):
        suchtext = st.text_input("🔎 Ticker-Suche (Filter innerhalb der Auswahl)", "", key="betrag_suche_global").lower().strip()
        for k, tickers in ausgewaehlte_aktien.items():
            filtered = [t for t in tickers if suchtext in t.lower()] if suchtext else tickers
            filtered = list(dict.fromkeys(filtered))
            if not filtered:
                continue

            st.markdown(f"### {k}")
            cols = st.columns(3)
            betraege[k] = {}
            for i, t in enumerate(filtered):
                with cols[i % 3]:
                    betraege[k][t] = st.number_input(
                        f"{t} – Betrag",
                        min_value=0,
                        step=100,
                        value=int(st.session_state.get(f"amt_{k}_{t}", 0)),
                        key=f"amt_{k}_{t}",
                    )

with col_right:
    st.subheader("📈 Aktuelle Portfolio-Zusammensetzung (nach Betrag)")
    sum_cat = {k: sum(betraege.get(k, {}).values()) for k in verteilung}
    total_amt = sum(sum_cat.values())

    if total_amt > 0:
        aktuelle = {k: round((sum_cat[k] / total_amt) * 100, 1) for k in verteilung}
        df_vergleich = pd.DataFrame(
            [
                {
                    "Kategorie": k,
                    "Empfohlen (%)": verteilung[k],
                    "Investiert (USD)": round(sum_cat[k], 2),
                    "Aktuell (%)": round(aktuelle.get(k, 0), 1),
                    "Differenz (%)": round(aktuelle.get(k, 0) - verteilung[k], 1),
                }
                for k in verteilung
            ]
        )
        st.dataframe(
            df_vergleich.style.format(
                {
                    "Investiert (USD)": "{:,.2f}",
                    "Empfohlen (%)": "{:.1f}",
                    "Aktuell (%)": "{:.1f}",
                    "Differenz (%)": "{:+.1f}",
                }
            ),
            use_container_width=True,
        )
        st.metric("Gesamt investiert", f"{total_amt:,.2f} USD")
    else:
        st.info("Noch keine Beträge erfasst.")

# ------------------------------------------------------------
# Portfolios verwalten (ES CRUD)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🗂 Portfolios verwalten")

begruendung = st.text_area("📝 Bemerkung: Auf was muss ich achten?")

col_a, col_b = st.columns([2, 2])
with col_a:
    portfolio_name = st.text_input("📛 Portfolio-Name", value=st.session_state.get("portfolio_name", ""))
with col_b:
    vorhandene = list_portfolios(es)
    labels = ["—"] + [
        f'{p["name"]} — {p.get("market_condition","")} — {p.get("totals",{}).get("total_amount",0):,.2f} USD'
        for p in vorhandene
    ]
    selected_label = st.selectbox("Vorhandenes Portfolio laden:", labels, index=0)

c1, c2, c3, c4 = st.columns(4)
with c1:
    load_clicked = st.button("📥 Laden")
with c2:
    save_new_clicked = st.button("💾 Neu speichern")
with c3:
    update_clicked = st.button("🔁 Aktualisieren")
with c4:
    delete_clicked = st.button("🗑️ Löschen")

# --- Laden ---
if load_clicked and selected_label != "—":
    idx = labels.index(selected_label) - 1
    p = vorhandene[idx]
    data = load_portfolio(es, p["id"])
    if data:
        gespeicherte_auswahl = {}
        gespeicherte_betraege = {}
        for it in data.get("items", []):
            gespeicherte_auswahl.setdefault(it["category"], []).append(it["symbol"])
            gespeicherte_betraege.setdefault(it["category"], {})[it["symbol"]] = float(it["amount"])

        st.session_state["hydrate_payload"] = {
            "portfolio_id": p["id"],
            "portfolio_name": data.get("name", ""),
            "auswahl": gespeicherte_auswahl,
            "betraege": gespeicherte_betraege,
        }
        st.session_state["current_portfolio_id"] = p["id"]
        st.success(f'Portfolio "{data.get("name")}" geladen.')
        st.rerun()
    else:
        st.error("Portfolio nicht gefunden.")

# --- Neu speichern ---
if save_new_clicked:
    if not portfolio_name.strip():
        st.warning("Bitte Portfolio-Name angeben.")
    else:
        doc = build_portfolio_doc(
            name=portfolio_name.strip(),
            market_condition=marktlage,
            selected_industry=selected_industry,
            comment=begruendung,
            selected_stocks=ausgewaehlte_aktien,
            amounts=betraege,
            allocation=verteilung,
        )

        new_id = save_portfolio(es, doc)

        st.session_state["hydrate_payload"] = {
            "portfolio_id": new_id,
            "portfolio_name": doc.get("name", ""),
            "auswahl": ausgewaehlte_aktien,
            "betraege": betraege,
        }

        st.session_state["widget_seed"] = str(uuid.uuid4())
        st.success(f'Portfolio "{doc.get("name","")}" gespeichert.')
        st.rerun()

# --- Aktualisieren ---
if update_clicked:
    pid = st.session_state.get("current_portfolio_id")
    if not pid:
        st.warning("Kein Portfolio geladen. Bitte zuerst laden oder neu speichern.")
    elif not portfolio_name.strip():
        st.warning("Bitte Portfolio-Name angeben.")
    else:
        doc = build_portfolio_doc(
            name=portfolio_name.strip(),
            market_condition=marktlage,
            selected_industry=selected_industry,
            comment=begruendung,
            selected_stocks=ausgewaehlte_aktien,
            amounts=betraege,
            allocation=verteilung,
        )
        save_portfolio(es, doc, portfolio_id=pid)
        st.success("Portfolio aktualisiert.")
        st.rerun()

# --- Löschen ---
if delete_clicked:
    pid = st.session_state.get("current_portfolio_id")
    if not pid:
        st.warning("Kein Portfolio geladen.")
    else:
        if delete_portfolio(es, pid):
            st.success("Portfolio gelöscht.")
            st.session_state["hydrate_payload"] = {
                "portfolio_id": None,
                "portfolio_name": "",
                "auswahl": {},
                "betraege": {},
            }
            st.session_state["current_portfolio_id"] = None
            st.rerun()
        else:
            st.error("Löschen fehlgeschlagen.")
