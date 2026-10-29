"""Per-horizon model selection.

Every horizon gets its own model: the model family, its complexity and the
Ridge/LightGBM blend are chosen separately for each horizon, by purged
expanding-window cross-validation on that horizon's own training period
(never on the test period). Selection criterion: mean fold IC (Spearman),
since the accuracy tables rank on direction and ordering.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

_GBM = {
    "small": dict(n_estimators=150, learning_rate=0.03, num_leaves=4, min_child_samples=100),
    "medium": dict(n_estimators=200, learning_rate=0.02, num_leaves=8, min_child_samples=100),
    "large": dict(n_estimators=500, learning_rate=0.02, num_leaves=16, min_child_samples=50),
}
_COMMON = dict(subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1)

# name -> (Ridge weight, LightGBM size or None)
CONFIGS = {
    "ridge": (1.0, None),
    "lgbm_small": (0.0, "small"),
    "lgbm_medium": (0.0, "medium"),
    "lgbm_large": (0.0, "large"),
    "blend_small": (0.5, "small"),
    "blend_medium": (0.5, "medium"),      # the shared default used before
    "blend_large": (0.5, "large"),
}
LABEL_ZH = {"ridge": "Ridge", "lgbm_small": "LightGBM 小", "lgbm_medium": "LightGBM 中",
            "lgbm_large": "LightGBM 大", "blend_small": "Ridge+LGBM 小", "blend_medium": "Ridge+LGBM 中",
            "blend_large": "Ridge+LGBM 大"}


class Fitted:
    def __init__(self, name, X, y):
        w, size = CONFIGS[name]
        lo, hi = y.quantile([0.01, 0.99])
        y = y.clip(lo, hi)
        self.keep = X.columns[X.notna().mean() > 0.5]
        self.w = w
        self.ridge = self.gbm = None
        if w > 0:
            self.ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       RidgeCV(alphas=np.logspace(1, 6, 21), cv=TimeSeriesSplit(5))
                                       ).fit(X[self.keep], y)
        if size is not None:
            self.gbm = lgb.LGBMRegressor(**_GBM[size], **_COMMON).fit(X[self.keep], y)

    def predict(self, X):
        Xk = X[self.keep]
        p = 0.0
        if self.ridge is not None:
            p = p + self.w * self.ridge.predict(Xk)
        if self.gbm is not None:
            p = p + (1 - self.w) * self.gbm.predict(Xk)
        return np.asarray(p)


def select(X, y, train_end, purge, n_folds=3, group=None):
    """Choose a config for one horizon using only rows before `train_end`.

    `group` is the date of each row (intraday: many rows per day), used so
    folds and purges fall on whole days. `purge` is in units of `group`."""
    group = pd.Series(X.index if group is None else group, index=X.index)
    ok = y.notna() & (group < train_end)
    ug = np.sort(group[ok].unique())
    edges = np.linspace(len(ug) * 0.4, len(ug), n_folds + 1).astype(int)
    scores = {k: [] for k in CONFIGS}
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = ug[a], ug[b - 1]
        cut = ug[max(a - purge, 0)]
        tr = ok & (group < cut)
        va = ok & (group >= lo) & (group <= hi)
        for name in CONFIGS:
            p = Fitted(name, X[tr], y[tr]).predict(X[va])
            scores[name].append(stats.spearmanr(p, y[va]).statistic)
    mean = {k: float(np.nanmean(v)) for k, v in scores.items()}
    return max(mean, key=mean.get), mean
