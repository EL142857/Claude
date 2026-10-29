"""Intraday NVDA forecasts: from 1 hour down to 1 minute, next-bar returns.

Same idea as the daily model (Ridge + LightGBM ensemble, walk-forward refits,
calibrated risk bands), on intraday bars from Yahoo. Yahoo's finest bar is 1
minute, so horizons below that (e.g. 30 seconds) need a tick-data source.

Intraday bars are cached and *appended* on every run, so the history grows
beyond Yahoo's rolling window (60 days for 5m, ~30 days for 1m) if this is
re-run regularly.
"""
import time

import numpy as np
import pandas as pd

from data import RAW, _safe, load_earnings_for
from model import _fit_return_models

TICKERS = ["NVDA", "QQQ", "SMH", "SPY", "^VIX"]
# interval -> (minutes, yfinance period or None for chunked 1m)
INTERVALS = {"60m": (60, "730d"), "30m": (30, "60d"), "15m": (15, "60d"),
             "5m": (5, "60d"), "2m": (2, "60d"), "1m": (1, None)}
N_CHUNKS = 10          # walk-forward refits over the test half


def _download(ticker, interval):
    import yfinance as yf

    if INTERVALS[interval][1] is not None:
        frames = [yf.download(ticker, period=INTERVALS[interval][1], interval=interval,
                              progress=False, auto_adjust=True, prepost=False)]
    else:  # 1m: Yahoo serves the last ~30 days in windows of at most 8 days
        end = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(days=1)
        frames = []
        for k in range(4):
            e, s = end - pd.Timedelta(days=7 * k), end - pd.Timedelta(days=7 * (k + 1))
            frames.append(yf.download(ticker, start=s.date(), end=e.date(), interval="1m",
                                      progress=False, auto_adjust=True, prepost=False))
            time.sleep(1)
    df = pd.concat([f for f in frames if len(f)])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load(ticker, interval, refresh=True):
    """Cached bars, merged with a fresh download (new bars appended)."""
    path = RAW / f"intra_{interval}_{_safe(ticker)}.csv"
    old = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    if refresh or old is None:
        new = _download(ticker, interval)
        df = new if old is None else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(path)
    else:
        df = old
    # drop today's bars while the session is still open (last bar incomplete)
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    if now.hour < 17:
        df = df[df.index < now.normalize()]
    return df


def _third_friday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21


def build(interval, daily_mu=None, refresh=True):
    """Features at each bar close and the next-bar log return as target.

    Rows whose next bar is in another session, or is a shortened bar (the
    15:30 60m bar only lasts 30 minutes), are not used as targets."""
    mins = INTERVALS[interval][0]
    bars = {t: load(t, interval, refresh) for t in TICKERS}
    nv = bars["NVDA"]
    idx = nv.index
    c = nv["Close"]
    close = {t: b["Close"].reindex(idx, method="ffill") for t, b in bars.items()}
    date = pd.Series(idx.normalize(), index=idx)
    lr = np.log(c).diff()
    same_day = date.eq(date.shift(1))
    r_in = lr.where(same_day)                    # intraday bar returns only
    F = {}

    for k in (1, 2, 3, 6, 12):
        F[f"ret{k}"] = np.log(c / c.shift(k)).where(date.eq(date.shift(k)))
    day_open = nv["Open"].groupby(date).transform("first")
    prev_close = c.groupby(date).last().shift(1).reindex(date.values).set_axis(idx)
    F["ret_since_open"] = np.log(c / day_open)
    F["gap_today"] = np.log(day_open / prev_close)
    F["ret_since_prev_close"] = np.log(c / prev_close)
    for t in ("QQQ", "SMH", "SPY"):
        s = np.log(close[t])
        for k in (1, 3):
            F[f"{t}_ret{k}"] = (s - s.shift(k)).where(date.eq(date.shift(k)))
        F[f"{t}_since_open"] = s - s.groupby(date).transform("first")
    q1 = np.log(close["QQQ"]).diff().where(same_day)
    beta = r_in.rolling(500, min_periods=100).cov(q1) / q1.rolling(500, min_periods=100).var()
    resid = r_in - beta * q1
    rs = resid.rolling(200, min_periods=50).std()
    for k in (1, 3, 12):
        F[f"resid{k}_z"] = resid.rolling(k).sum() / (rs * np.sqrt(k))     # stat-arb pressure
    v = np.log(close["^VIX"])
    F["vix_chg1"], F["vix_chg6"] = v.diff(), v.diff(6)
    F["vix_lvl"] = close["^VIX"]

    # volume and range, against the same time-of-day slot (intraday U-shape)
    slot = (idx.hour * 60 + idx.minute - 570) // mins
    F["slot"] = pd.Series(slot, index=idx).astype(float)
    lv = np.log(nv["Volume"].replace(0, np.nan))
    lv_norm = lv - lv.groupby(slot).transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))
    F["volume_z"] = lv_norm
    F["range"] = np.log(nv["High"] / nv["Low"])
    F["close_loc"] = ((c - nv["Low"]) / (nv["High"] - nv["Low"])).where(nv["High"] > nv["Low"], 0.5)
    F["rv12"] = r_in.rolling(12, min_periods=6).std()
    F["rv_day"] = r_in.groupby(date).transform(lambda x: x.expanding().std())

    # calendar / mechanical flows
    mod = idx.hour * 60 + idx.minute
    end_min = np.minimum(mod + mins, 960)                    # bar end, capped at 16:00
    F["last30"] = pd.Series((end_min >= 930).astype(float), index=idx)   # MOC / leveraged-ETF rebalancing
    F["first30"] = pd.Series((mod < 600).astype(float), index=idx)
    first_bar = F["ret_since_prev_close"].groupby(date).transform("first")
    F["intraday_mom"] = first_bar * F["last30"]              # opening move vs. the close (intraday momentum)
    F["dow"] = pd.Series(idx.dayofweek, index=idx).astype(float)
    F["opex_day"] = pd.Series([float(_third_friday(d)) for d in idx], index=idx)
    try:
        e = load_earnings_for("NVDA")
        react = set((pd.DatetimeIndex(e) + pd.offsets.BDay(1)).normalize())
        F["earnings_reaction_day"] = date.isin(react).astype(float)
    except Exception:
        pass
    if daily_mu is not None:
        # daily external-factor model's forecast made at the previous close for today
        F["daily_model_mu"] = date.map(daily_mu.shift(1).reindex(sorted(set(date.values))).to_dict())

    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
    nxt_same = date.shift(-1).eq(date)
    nxt_start = pd.Series(idx, index=idx).shift(-1)
    nxt_full = ((np.minimum(nxt_start.dt.hour * 60 + nxt_start.dt.minute + mins, 960)
                 - (nxt_start.dt.hour * 60 + nxt_start.dt.minute)) == mins)
    y = np.log(c.shift(-1) / c).where(nxt_same & nxt_full)
    return X, y, c, pd.Series(slot, index=idx)


def walk_forward(X, y, slot, test_frac=0.5):
    """Expanding-window refits at N_CHUNKS date boundaries over the test half.

    Risk band: realised vol of de-seasonalised bar returns, times the next
    slot's typical size, calibrated with training quantiles."""
    dates = X.index.normalize().unique().sort_values()
    test_dates = dates[int(len(dates) * (1 - test_frac)):]
    chunks = np.array_split(np.asarray(test_dates), N_CHUNKS)
    r = X["ret1"]                                   # past bar returns (NaN at each day's first bar)
    out = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        start, stop = ch[0], ch[-1] + pd.Timedelta(days=1)
        tr = (X.index < start) & y.notna().values
        te = (X.index >= start) & (X.index < stop)
        keep, ridge, gbm = _fit_return_models(X[tr], y[tr])
        # seasonal size of a bar by slot, from training only
        s_slot = y[tr].abs().groupby(slot[tr]).mean()
        s_all = y[tr].abs().mean()
        seas = (s_slot / s_all).reindex(range(int(slot.max()) + 2)).fillna(1.0)
        r_ds = r / seas.reindex(slot.values).set_axis(r.index).values
        rv = np.sqrt((r_ds ** 2).ewm(span=60, min_periods=20).mean())
        nxt = (slot + 1).clip(upper=seas.index.max())
        sigma = rv * seas.reindex(nxt.values).set_axis(nxt.index).values
        z = ((y[tr] - y[tr].mean()) / sigma[tr]).dropna()
        q = np.nanquantile(z, [0.025, 0.10, 0.90, 0.975])
        Xb = X.loc[te, keep]
        pr, pg = ridge.predict(Xb), gbm.predict(Xb)
        f = pd.DataFrame({"mu": 0.5 * (pr + pg), "sigma": sigma[te].values,
                          "train_mean": y[tr].mean()}, index=X.index[te])
        for name, qq in zip(("band_q025", "band_q10", "band_q90", "band_q975"), q):
            f[name] = f["mu"] + qq * f["sigma"]
        out.append(f)
    res = pd.concat(out)
    res["y"] = y.reindex(res.index)
    return res.dropna(subset=["y"])
