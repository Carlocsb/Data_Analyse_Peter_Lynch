
import streamlit as st
import os, sys, math
from datetime import datetime
import pandas as pd
import streamlit as st
import importlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Aktienanalyse nach Peter Lynch", layout="wide")

# Seitenleiste
st.sidebar.image("assets/Logo-TH-Köln1.png", caption="")
st.sidebar.info("Willkommen im Aktienanalyse-Dashboard von Carlo Beck.")

# Startseite
st.title("Willkommen zum Analyse-Dashboard nach der Peter Lynch Strategie")
st.markdown("""
Dieses Projekt dient der Analyse und Bewertung von Aktien gemäß den sechs Kategorien der **Peter Lynch Strategie**.  
Es basiert auf einer Datenbankanbindung und bietet folgende Funktionen:

- 🔍 Aktiensuche mit Kennzahlenübersicht  
- 📊 Automatische Kategorisierung nach Lynch  
- 📈 Visualisierung der Top 10 je Kategorie  
- 📁 Portfolio-Zusammenstellung mit Begründung
""")
