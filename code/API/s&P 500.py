from pathlib import Path
import pandas as pd
import yfinance as yf

START_DATE = "2010-01-01"
TICKER = "^GSPC"  # S&P 500 Index
OUTFILE = Path("sp500_monthly_open_close_since_2010-01.csv")

def download_sp500_monthly_open_close(start=START_DATE, ticker=TICKER) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        interval="1mo",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",   # hilft oft, MultiIndex zu vermeiden
    )

    if df is None or df.empty:
        raise RuntimeError("Keine Daten erhalten. Teste alternativ ticker='SPY'.")

    df = df.reset_index()

    # 1) Falls MultiIndex-Spalten existieren: flatten
    # Beispiel: ('Open', '^GSPC') -> 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # 2) Monatslabel
    df["month"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)

    # 3) Selektieren & umbenennen
    out = df.loc[:, ["month", "Open", "Close"]].copy()
    out = out.rename(columns={"Open": "open", "Close": "close"})

    # 4) Sicherstellen, dass es echte Series sind (nicht 2D)
    out["open"] = pd.to_numeric(out["open"].squeeze(), errors="coerce")
    out["close"] = pd.to_numeric(out["close"].squeeze(), errors="coerce")

    out = out.dropna(subset=["open", "close"]).drop_duplicates(subset=["month"]).reset_index(drop=True)
    return out

if __name__ == "__main__":
    data = download_sp500_monthly_open_close()
    data.to_csv(OUTFILE, index=False)
    print(f"Gespeichert: {OUTFILE.resolve()}")
    print(data.head(3))
    print("...")
    print(data.tail(3))
