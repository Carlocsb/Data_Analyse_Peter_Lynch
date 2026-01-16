CATEGORIES = {
    "Slow Growers": [
        ("earningsGrowth", "Gewinnwachstum < 5 %", lambda x: x < 0.05, False),
        ("dividendYield", "Dividendenrendite 3–9 %", lambda x: 0.03 <= x <= 0.09, False),
        ("payoutRatio", "Payout Ratio < 80 %", lambda x: x < 0.80, False),
        ("revenueGrowth", "Umsatzwachstum < 5 %", lambda x: x < 0.05, False),
        ("trailingPE", "KGV niedrig (<15)", lambda x: x < 15, True),
        ("debtToAssets", "Debt/Assets < 0.5", lambda x: x < 0.5, True),
    ],

    "Stalwarts": [
        ("earningsGrowth", "Gewinnwachstum 5–10 %", lambda x: 0.05 <= x <= 0.10, False),
        ("dividendYield", "Dividendenrendite ≥ 2 %", lambda x: x >= 0.02, False),
        ("trailingPE", "KGV < 25", lambda x: x < 25, False),
        ("marketCap", "Marktkapitalisierung > 10 Mrd", lambda x: x >= 10e9, False),
        ("freeCashFlow", "Free Cash Flow > 0", lambda x: x > 0, False),
        ("debtToAssets", "Debt/Assets < 0.5", lambda x: x < 0.5, True),
    ],

    "Fast Growers": [
        ("earningsGrowth", "Gewinnwachstum > 20 %", lambda x: x > 0.20, False),
        ("revenueGrowth", "Umsatzwachstum > 20 %", lambda x: x > 0.20, False),
        ("trailingPE", "KGV < 25", lambda x: x < 25, False),
        ("pegRatio", "PEG < 1", lambda x: x < 1, True),
        ("debtToAssets", "Debt/Assets niedrig (<0.5)", lambda x: x < 0.5, True),
        ("priceToBook", "P/B moderat (<4)", lambda x: x < 4, True),
    ],

    "Cyclicals": [
        ("sector", "Zyklischer Sektor", lambda s: str(s).lower() in {"auto", "automotive", "stahl", "steel", "bau", "construction", "chemicals", "metals", "airlines", "travel", "energy", "basic materials"}, False),
        ("revenueGrowth", "Umsatz erholt sich (>5 %)", lambda x: x > 0.05, False),
        ("epsGrowth", "Gewinne steigen (>0 %)", lambda x: x > 0.0, False),
        ("trailingPE", "KGV < 25", lambda x: x < 25, False),
        ("freeCashFlowPerShare", "FCF/Aktie > 0", lambda x: x > 0, False),
        ("debtToAssets", "Debt/Assets < 0.5", lambda x: x < 0.5, True),
    ],

    "Turnarounds": [
        ("cashToDebt", "Cash ≥ 50 % der Schulden", lambda x: x >= 0.5, False),                # = totalCash/totalDebt
        ("equityRatio", "Eigenkapitalquote > 30 %", lambda x: x > 0.30, False),               # = totalEquity/totalAssets
        ("fcfMargin", "FCF-Marge ≥ 5 %", lambda x: x >= 0.05, False),                         # = freeCashFlow/revenue
        ("revenueGrowth", "Umsatz wieder steigend (>0 %)", lambda x: x > 0.0, False),
        ("sgaTrend", "SG&A-Quote rückläufig", lambda x: x is True, True),
        ("currentRatio", "Liquidität (Current Ratio ≥ 1)",lambda x: x >= 1.0,   False),                   
    ],

    "Asset Plays": [
        ("priceToBook", "P/B < 1", lambda x: x < 1.0, False),
        ("bookValuePerShare", "Substanz (Buchwert je Aktie vorhanden)", lambda x: x is not None and x > 0, False),
        ("cashPerShare", "Cash je Aktie > 5 $", lambda x: x > 5, False),
        ("trailingPE", "KGV < 20", lambda x: x < 20, True),
        ("debtToAssets", "Debt/Assets < 0.5", lambda x: x < 0.5, True),
         ("marketCap", "eher kleiner ( < 10 Mrd USD )", lambda x: x < 10e9, True),],
        # Qualitatives Kriterium „versteckte Werte“ kann nur als manuelle Notiz/Flag gepflegt werden.
    
}
