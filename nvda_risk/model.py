"""Walk-forward forecast ensemble with a risk-control layer.

For each horizon h:
  1. Return model  : Ridge (linear factor exposures) + LightGBM (non-linear,
                     regime interactions), averaged.
  2. Risk model    : HAR regression of forward realised vol on past vol + VIX,
                     plus empirical quantiles of standardised errors, which
                     give calibrated 80% / 95% price bands.
  3. Risk control  : position = forecast / variance, halved when VIX is in
                     backwardation, capped at +/-1.

Training is strictly walk-forward. Labels overlap for h > 1, so every fit drops
the last h rows before the refit date (purge) to keep test labels unseen.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MIN_TRAIN, RETRAIN_EVERY

warnings.filterwarnings("ignore", category=UserWarning)

LGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.02, num_leaves=8, min_child_samples=100,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
    verbose=-1,
)


def _fit_return_models(X, y):
    lo, hi = y.quantile([0.01, 0.99])
    y = y.clip(lo, hi)
    keep = X.columns[X.notna().mean() > 0.5]
    ridge = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        RidgeCV(alphas=np.logspace(1, 6, 21), cv=TimeSeriesSplit(5)),  # LOO-CV is biased on overlapping labels
    ).fit(X[keep], y)
    gbm = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[keep], y)
    return keep, ridge, gbm


def _vol_design(X, r1):
    """HAR-RV inputs: daily, weekly, monthly realised vol, and VIX if present."""
    d = pd.DataFrame({
        "rv1": np.log(r1.abs() + 1e-4),
        "rv5": np.log(r1.rolling(5).std() + 1e-4),
        "rv22": np.log(r1.rolling(22).std() + 1e-4),
    })
    if "^VIX_lvl_z" in X:
        d["vix_z"] = X["^VIX_lvl_z"]
    return d


def walk_forward(X, y, r1, test_start, h, cols=None, config=None):
    """Return a DataFrame of out-of-sample forecasts for horizon h."""
    cols = X.columns if cols is None else cols
    X = X[cols]
    fwd_vol = np.log(r1[::-1].rolling(h).std()[::-1].shift(-1) * np.sqrt(h) + 1e-4) if h > 1 \
        else np.log(r1.shift(-1).abs() + 1e-4)
    V = _vol_design(X, r1)
    x_ok = X.notna().mean(axis=1).gt(0.5)
    valid = y.notna() & x_ok
    # forecast every usable date, including the latest ones whose outcome is not known yet
    dates = X.index[(X.index >= test_start) & x_ok]
    if len(dates) == 0:
        return pd.DataFrame()
    out = []
    for k in range(0, len(dates), RETRAIN_EVERY):
        block = dates[k:k + RETRAIN_EVERY]
        cut = X.index.get_loc(block[0]) - h          # purge overlapping labels
        tr = X.index[:max(cut, 0)]
        tr = tr[valid.reindex(tr).values]
        if len(tr) < MIN_TRAIN:
            continue
        if config is None:
            keep, ridge, gbm = _fit_return_models(X.loc[tr], y.loc[tr])
        else:                                            # per-horizon model chosen by tuning.select
            import tuning
            fitted = tuning.Fitted(config, X.loc[tr], y.loc[tr])
        Vtr = V.loc[tr].dropna()
        vtr_y = fwd_vol.reindex(Vtr.index)
        ok = vtr_y.notna()
        vol_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 11))).fit(
            Vtr[ok], vtr_y[ok])

        # calibrate band width on standardised returns around the training mean:
        # in-sample model residuals are too small (overfit) and give narrow bands
        sig_tr = np.exp(vol_model.predict(V.loc[tr].fillna(V.loc[tr].median())))
        z = (y.loc[tr].values - y.loc[tr].mean()) / sig_tr
        q = np.nanquantile(z, [0.025, 0.10, 0.90, 0.975])

        if config is None:
            Xb = X.loc[block, keep]
            pr, pg = ridge.predict(Xb), gbm.predict(Xb)
            mu = 0.5 * (pr + pg)
        else:
            mu = fitted.predict(X.loc[block])
            pr = pg = np.full(len(block), np.nan)
        sig = np.exp(vol_model.predict(V.loc[block].fillna(V.loc[tr].median())))
        out.append(pd.DataFrame({
            "ridge": pr, "lgbm": pg, "mu": mu, "sigma": sig,
            "q025": q[0], "q10": q[1], "q90": q[2], "q975": q[3],
            "train_mean": y.loc[tr].mean(),
        }, index=block))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out)
    res["y"] = y.reindex(res.index)
    for qn in ("q025", "q10", "q90", "q975"):
        res[f"band_{qn}"] = res["mu"] + res[qn] * res["sigma"]
    return res


def risk_controlled_position(fc, X, h, kelly=0.5):
    """Fractional-Kelly sizing on mu / sigma^2, stress-halved, capped at +/-1."""
    mu_d, var_d = fc["mu"] / h, (fc["sigma"] ** 2) / h
    pos = (kelly * mu_d / var_d).clip(-1, 1)
    if "vix_term" in X:
        stress = X["vix_term"].reindex(fc.index) > 1.0   # VIX curve inverted
        pos = pos.where(~stress, pos * 0.5)
    return pos
