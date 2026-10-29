"""Data loading with a local CSV cache.

Each series is cached under data/raw/. If a file is already there it is used
as-is, so you can also drop in CSVs downloaded elsewhere:
  yahoo_<TICKER>.csv : Date,Open,High,Low,Close,Volume (adjusted prices)
  fred_<SERIES>.csv  : Date,<SERIES>
"""
import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

from config import EARNINGS_FALLBACK, FRED, START, YAHOO

RAW = Path(__file__).parent / "data" / "raw"


def _safe(name):
    return name.replace("^", "").replace("=", "_").replace(".", "_")


def _yahoo_one(ticker, start):
    import yfinance as yf

    for attempt in range(3):
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if len(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        time.sleep(2 ** attempt)
    raise RuntimeError(f"no Yahoo data for {ticker}")


def _fred_one(series, start):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
    with urllib.request.urlopen(url, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()), na_values=".")
    df.columns = ["Date", series]
    return df.set_index("Date")


def _cached(path, fetch):
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_all(start=START, refresh=False):
    """Return (ohlcv dict, fred DataFrame, earnings dates, missing list)."""
    if refresh:
        for f in RAW.glob("*.csv"):
            f.unlink()
    ohlcv, missing = {}, []
    for t in YAHOO:
        try:
            ohlcv[t] = _cached(RAW / f"yahoo_{_safe(t)}.csv", lambda t=t: _yahoo_one(t, start))
        except Exception as e:  # keep going: one missing factor should not kill the run
            missing.append((t, str(e)[:120]))
    if "NVDA" not in ohlcv:
        raise RuntimeError(
            "NVDA price data is unavailable. Allow query1/query2.finance.yahoo.com "
            "in the network settings, or place data/raw/yahoo_NVDA.csv by hand."
        )

    fred = {}
    for s in FRED:
        try:
            fred[s] = _cached(RAW / f"fred_{s}.csv", lambda s=s: _fred_one(s, start))[s]
        except Exception as e:
            missing.append((s, str(e)[:120]))
    fred = pd.DataFrame(fred)

    return ohlcv, fred, load_earnings(), missing


def load_earnings():
    dates = set(pd.to_datetime(EARNINGS_FALLBACK))
    try:
        import yfinance as yf

        ed = yf.Ticker("NVDA").get_earnings_dates(limit=60)
        if ed is not None and len(ed):
            dates |= set(ed.index.tz_localize(None).normalize())
    except Exception:
        pass
    return pd.DatetimeIndex(sorted(dates))
