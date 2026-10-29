"""Compare forecasts with what actually happened."""
import numpy as np
import pandas as pd
from scipy import stats

COST_BP = 5  # per unit of turnover


def forecast_metrics(fc, price, h):
    """Return accuracy, ranking and calibration metrics for one forecast set."""
    y, mu = fc["y"], fc["mu"]
    m = y.notna()
    y, mu, f = y[m], mu[m], fc[m]
    sse_model = ((y - mu) ** 2).sum()
    sse_rw = (y ** 2).sum()
    sse_drift = ((y - f["train_mean"]) ** 2).sum()

    p0 = price.reindex(y.index)
    p_true = p0 * np.exp(y)
    p_hat = p0 * np.exp(mu)
    hits = np.sign(mu) == np.sign(y)
    n = int((y != 0).sum())
    k = int(hits[y != 0].sum())
    ic = stats.spearmanr(mu, y).statistic
    monthly_ic = pd.DataFrame({"mu": mu, "y": y}).groupby(y.index.to_period("M")).apply(
        lambda d: stats.spearmanr(d["mu"], d["y"]).statistic if len(d) > 5 else np.nan).dropna()

    def cover(lo, hi):
        return float(((y >= f[lo]) & (y <= f[hi])).mean())

    return {
        "horizon": h,
        "n": int(len(y)),
        "oos_r2_vs_random_walk": float(1 - sse_model / sse_rw),
        "oos_r2_vs_drift": float(1 - sse_model / sse_drift),
        "price_mape_model": float((abs(p_hat - p_true) / p_true).mean()),
        "price_mape_random_walk": float((abs(p0 - p_true) / p_true).mean()),
        "hit_rate": k / n if n else np.nan,
        "hit_rate_pvalue": float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else np.nan,
        "up_share_actual": float((y > 0).mean()),
        # the bar a direction call must clear: always betting on the majority side
        "hit_rate_always_up": float(max((y > 0).mean(), (y < 0).mean())),
        "hit_rate_pvalue_vs_always_up": float(stats.binomtest(
            k, n, max((y > 0).mean(), (y < 0).mean()), alternative="greater").pvalue) if n else np.nan,
        "share_pred_up": float((mu > 0).mean()),
        "ic_spearman": float(ic),
        "monthly_ic_mean": float(monthly_ic.mean()),
        "monthly_ic_tstat": float(monthly_ic.mean() / monthly_ic.std() * np.sqrt(len(monthly_ic)))
        if len(monthly_ic) > 2 else np.nan,
        "coverage_80": cover("band_q10", "band_q90"),
        "coverage_95": cover("band_q025", "band_q975"),
    }


def strategy(pos, r_next):
    """Daily P&L of holding `pos` from close t to close t+1, after costs."""
    r = (np.exp(r_next) - 1).reindex(pos.index)
    m = r.notna()
    pos, r = pos[m], r[m]
    pnl = pos * r - pos.diff().abs().fillna(pos.abs()) * COST_BP / 1e4
    return pnl, r


def perf(pnl):
    eq = (1 + pnl).cumprod()
    yrs = len(pnl) / 252
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
        "ann_vol": float(pnl.std() * np.sqrt(252)),
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }
