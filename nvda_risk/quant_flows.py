"""Quantified proxies for what systematic / quant funds are doing.

We cannot see fund positions directly, so each function rebuilds the trading
rule a given class of fund is known to follow from public prices. Its output
is that fund class's *estimated position* and *estimated flow* (change in
position), which is the part that moves the price.

All functions use data up to and including day t only.
"""
import numpy as np
import pandas as pd

ANN = np.sqrt(252)


def ewm_vol(r, span=60):
    return r.ewm(span=span, min_periods=span // 2).std() * ANN


def cta_trend(close, lookbacks=(20, 60, 120, 250), target_vol=0.15, cap=2.0):
    """Trend-following CTA replica (SG Trend style).

    Signal per lookback = tanh of the vol-normalised trailing return; position =
    mean signal scaled to a vol target. Returns position and 5-day flow.
    """
    r = np.log(close).diff()
    vol = ewm_vol(r)
    daily_vol = vol / ANN
    sigs = []
    for lb in lookbacks:
        z = np.log(close / close.shift(lb)) / (daily_vol * np.sqrt(lb))
        sigs.append(np.tanh(z))
    pos = (pd.concat(sigs, axis=1).mean(axis=1) * target_vol / vol).clip(-cap, cap)
    return pd.DataFrame({"pos": pos, "flow5": pos.diff(5)})


def vol_control(index_close, target_vol=0.10, cap=1.5):
    """Volatility-targeting funds / variable annuity hedgers.

    Equity exposure = target vol / max(short, long realised vol). When vol jumps
    they sell mechanically, regardless of fundamentals.
    """
    r = np.log(index_close).diff()
    rv = pd.concat([r.rolling(20).std(), r.rolling(60).std()], axis=1).max(axis=1) * ANN
    expo = (target_vol / rv).clip(upper=cap)
    return pd.DataFrame({"expo": expo, "flow5": expo.diff(5)})


def risk_parity(stock_close, bond_close, window=60, target_vol=0.10):
    """Risk-parity funds: leverage falls when stocks and bonds sell off together.

    Uses a 50/50 risk-weighted stock/bond book; its realised vol drives leverage.
    """
    rs, rb = np.log(stock_close).diff(), np.log(bond_close).diff()
    vs, vb = rs.rolling(window).std(), rb.rolling(window).std()
    ws, wb = (1 / vs) / (1 / vs + 1 / vb), (1 / vb) / (1 / vs + 1 / vb)
    port_vol = (ws * rs + wb * rb).rolling(window).std() * ANN
    lev = (target_vol / port_vol).clip(upper=4)
    corr = rs.rolling(window).corr(rb)
    return pd.DataFrame({"lev": lev, "flow5": lev.diff(5), "sb_corr": corr})


def leveraged_etf_rebalance(nvda_close, nvdl_close, nvda_vol, nvdl_vol, L=2.0):
    """Leveraged ETFs must buy L*(L-1)*r of their AUM at the close every day.

    AUM is not in the price feed, so we scale by NVDL's dollar volume relative
    to NVDA's as a size proxy. Zero before NVDL existed.
    """
    r = nvda_close.pct_change()
    if nvdl_close is None:
        return pd.Series(0.0, index=nvda_close.index)
    size = (nvdl_close * nvdl_vol).rolling(20).mean() / (nvda_close * nvda_vol).rolling(20).mean()
    return (L * (L - 1) * r * size.reindex(r.index)).fillna(0.0)


def momentum_crowding(mtum, usmv, target):
    """Momentum-factor crowding: how much NVDA trades as a 'momentum stock',
    and whether the momentum factor itself is crashing (de-grossing)."""
    rm, rv, rt = (np.log(x).diff() for x in (mtum, usmv, target))
    spread = rm - rv
    beta = rt.rolling(60).cov(spread) / spread.rolling(60).var()
    dd = mtum / mtum.rolling(60).max() - 1
    return pd.DataFrame({
        "mom_spread20": spread.rolling(20).sum(),
        "nvda_mom_beta": beta,
        "mom_drawdown": dd,
    })


def stat_arb_residual(target, sector, window=60):
    """Statistical arbitrage: stat-arb books fade NVDA's move relative to its
    sector. A large residual is what they sell (or buy back)."""
    rt, rs = np.log(target).diff(), np.log(sector).diff()
    beta = rt.rolling(window).cov(rs) / rs.rolling(window).var()
    resid = rt - beta * rs
    sd = resid.rolling(window).std()
    return pd.DataFrame({
        "resid5_z": resid.rolling(5).sum() / (sd * np.sqrt(5)),
        "resid20_z": resid.rolling(20).sum() / (sd * np.sqrt(20)),
        "sector_beta": beta,
    })


def calendar_flows(index):
    """Calendar-driven mechanical flows: option expiry (dealer gamma unwinds),
    quad witching / index rebalances, and month/quarter-end pension rebalancing."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    third_fri = (s.dt.weekday == 4) & (s.dt.day.between(15, 21))
    opex = pd.Series(third_fri.values, index=idx)
    # days until the next monthly opex (in trading days)
    pos = np.arange(len(idx))
    nxt = pd.Series(np.where(opex, pos, np.nan), index=idx).bfill()
    days_to_opex = (nxt - pos).fillna(20)
    quad = opex & idx.month.isin([3, 6, 9, 12])
    month = idx.to_period("M")
    last_n = pd.Series(1, index=idx).groupby(month).cumcount(ascending=False)
    return pd.DataFrame({
        "days_to_opex": days_to_opex.values,
        "opex_week": (days_to_opex <= 4).astype(int).values,
        "quad_witching_week": (pd.Series(quad, index=idx).rolling(5).max().shift(-4).fillna(0)).values,
        "month_end": (last_n.values < 3).astype(int),
        "quarter_end": ((last_n.values < 3) & idx.month.isin([3, 6, 9, 12])).astype(int),
    }, index=idx)


def microstructure(o, h, l, c, v):
    """Who is trading: overnight vs intraday returns (intraday moves are
    dominated by institutions, overnight by news/retail), volume shock,
    Amihud illiquidity and range-based vol."""
    overnight = np.log(o / c.shift(1))
    intraday = np.log(c / o)
    dollar = c * v
    amihud = (np.log(c).diff().abs() / dollar).rolling(20).mean() * 1e9
    vol_z = (np.log(v) - np.log(v).rolling(60).mean()) / np.log(v).rolling(60).std()
    parkinson = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2))) * ANN
    return pd.DataFrame({
        "overnight20": overnight.rolling(20).sum(),
        "intraday20": intraday.rolling(20).sum(),
        "volume_z": vol_z,
        "amihud": np.log(amihud),
        "parkinson_vol": parkinson,
    })
