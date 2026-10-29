"""One pooled LightGBM over all stocks, plus the tools to interrogate it.

- walk_forward : out-of-sample forecasts, refit every RETRAIN_EVERY dates
- fit_final    : one model on every labelled row (used for the analysis)
- shap_shares  : per stock, which factor groups drive the forecast
- scenarios    : shock a set of factors and read each stock's predicted reaction
- empirical    : the same reaction measured directly in history (HAC t-stats),
                 as a check that the model's differences between stocks are real
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import RETRAIN_EVERY, SCENARIOS

PARAMS = dict(
    n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, verbose=-1,
)
MIN_TRAIN_DATES = 750


def _fit(X, y):
    lo, hi = y.quantile([0.01, 0.99])
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X, y.clip(lo, hi), categorical_feature=["stock_id"])
    return m


def _dates(X):
    return X.index.get_level_values("Date")


def walk_forward(X, y_norm, scale, y_raw, test_start, h):
    d = _dates(X)
    udates = d.unique().sort_values()
    labelled = y_norm.notna()
    test_dates = udates[udates >= test_start]
    out = []
    for k in range(0, len(test_dates), RETRAIN_EVERY):
        block = test_dates[k:k + RETRAIN_EVERY]
        cut = udates[max(udates.get_loc(block[0]) - h, 0)]   # purge overlapping labels
        tr = (d < cut) & labelled
        if d[tr].nunique() < MIN_TRAIN_DATES:
            continue
        m = _fit(X[tr], y_norm[tr])
        z = (y_norm[tr] - y_norm[tr].mean()).values
        q = np.nanquantile(z, [0.025, 0.10, 0.90, 0.975])
        te = d.isin(block)
        pn = m.predict(X[te])
        f = pd.DataFrame({"pred_norm": pn, "scale": scale[te].values}, index=X.index[te])
        f["mu"] = f["pred_norm"] * f["scale"]
        f["train_mean"] = (y_norm[tr].mean() * f["scale"]).values
        for name, qq in zip(("band_q025", "band_q10", "band_q90", "band_q975"), q):
            f[name] = f["mu"] + qq * f["scale"]
        out.append(f)
    res = pd.concat(out)
    res["y"] = y_raw.reindex(res.index)
    return res


def fit_final(X, y_norm, until=None):
    tr = y_norm.notna()
    if until is not None:
        tr &= _dates(X) < until
    return _fit(X[tr], y_norm[tr])


def shap_shares(model, X, groups):
    """Mean |SHAP| per (stock, factor group), as a share of that stock's total."""
    contrib = model.predict(X, pred_contrib=True)[:, :-1]          # drop the bias column
    a = pd.DataFrame(np.abs(contrib), columns=X.columns, index=X.index)
    by_group = a.T.groupby(groups.reindex(X.columns).values).sum().T
    per_stock = by_group.groupby(level="ticker").mean()
    return per_stock.div(per_stock.sum(axis=1), axis=0)


def _shocked(X, spec, sd):
    """Return (base, shocked) copies of X for one scenario spec."""
    base, shock = X.copy(), X.copy()
    for f, how in spec.items():
        if f not in X:
            continue
        if isinstance(how, tuple):                                     # ("set", a, b)
            base[f], shock[f] = how[1], how[2]
        else:
            shock[f] = shock[f] + how * sd[f]
    return base, shock


def scenarios(model, X, scale, sd):
    """Average change in predicted h-day return (in %) per stock, per scenario."""
    rows = {}
    for key, (_, spec) in SCENARIOS.items():
        base, shock = _shocked(X, spec, sd)
        delta = (model.predict(shock) - model.predict(base)) * scale.reindex(X.index).values
        rows[key] = pd.Series(delta, index=X.index).groupby(level="ticker").mean() * 100
    return pd.DataFrame(rows).T


def empirical(X, y_raw, sd, mean):
    """Historical check: regress each stock's forward return on the scenario's
    composite score (average of the standardised, signed features). Returns the
    beta (% return per 1 sd of the composite) and its Newey-West t-stat."""
    beta, tstat = {}, {}
    for key, (_, spec) in SCENARIOS.items():
        parts = []
        for f, how in spec.items():
            if f not in X:
                continue
            if isinstance(how, tuple):
                parts.append((X[f] == how[2]).astype(float))               # event dummy
            else:
                parts.append(how * (X[f] - mean[f]) / sd[f])
        s = pd.concat(parts, axis=1).mean(axis=1)
        b, t = {}, {}
        for tk, g in pd.DataFrame({"s": s, "y": y_raw}).dropna().groupby(level="ticker"):
            if isinstance(next(iter(spec.values())), tuple):
                x = g["s"]                                             # beta = event-window effect
            else:
                x = g["s"] / g["s"].std()
            fit = sm.OLS(g["y"].values * 100, sm.add_constant(x.values)).fit(
                cov_type="HAC", cov_kwds={"maxlags": 5})
            b[tk], t[tk] = fit.params[1], fit.tvalues[1]
        beta[key], tstat[key] = pd.Series(b), pd.Series(t)
    return pd.DataFrame(beta).T, pd.DataFrame(tstat).T
