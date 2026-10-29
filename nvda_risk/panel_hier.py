"""Hierarchical linear panel model: one slope per factor shared by all stocks,
plus a per-stock deviation from it, shrunk toward zero.

    y_norm[s,t] = a_s + sum_f (b_f + d_{s,f}) * z_f[s,t]

The shrinkage is a ridge penalty alpha * (|b|^2 + |d|^2 / c^2). It is fitted by
scaling the deviation columns by c, so c = 0 is "every stock reacts the same",
large c is "every stock is on its own", and the data choose c by purged
time-series cross-validation on pre-test data only.

Because it is linear, a factor shock's predicted effect on stock s is simply
(b_f + d_{s,f}) x shock, so the stock-specific part d is explicit and can be
bootstrapped.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import RETRAIN_EVERY, SCENARIOS, UNIVERSE

ALPHAS = np.logspace(3, 7, 9)
CS = [0.0, 0.1, 0.25, 0.5, 1.0, 3.0]


class Hier:
    def __init__(self, alpha, c):
        self.alpha, self.c = alpha, c

    def _design(self, X, fit):
        F = X.drop(columns="stock_id")
        if fit:
            self.cols = F.columns
            self.med = F.median().fillna(0.0)            # a feature with no history yet is all zeros
            self.sd = F.fillna(self.med).std().replace(0, 1)
        Z = ((F[self.cols].fillna(self.med) - self.med) / self.sd).clip(-6, 6).to_numpy(np.float32)
        sid = X["stock_id"].to_numpy()
        D = np.zeros((len(X), len(UNIVERSE)), np.float32)
        D[np.arange(len(X)), sid] = 1.0
        blocks = [Z, D]                                   # common slopes + stock intercepts
        if self.c > 0:
            for s in range(len(UNIVERSE)):
                blocks.append(Z * D[:, [s]] * self.c)    # stock deviations, scaled for shrinkage
        return np.hstack(blocks)

    def fit(self, X, y):
        lo, hi = y.quantile([0.01, 0.99])
        self.m = Ridge(alpha=self.alpha).fit(self._design(X, True), y.clip(lo, hi).to_numpy())
        k = len(self.cols)
        w = self.m.coef_
        self.common = pd.Series(w[:k], index=self.cols)
        self.intercept = pd.Series(w[k:k + len(UNIVERSE)], index=UNIVERSE)
        if self.c > 0:
            dev = w[k + len(UNIVERSE):].reshape(len(UNIVERSE), k) * self.c
        else:
            dev = np.zeros((len(UNIVERSE), k))
        self.dev = pd.DataFrame(dev, index=UNIVERSE, columns=self.cols)
        return self

    def predict(self, X):
        return self.m.predict(self._design(X, False))

    def slopes(self):
        """Total slope per stock and feature, per 1 training-sd move of the feature."""
        return self.dev.add(self.common, axis=1)


class Separate:
    """No pooling at all: an independent ridge per stock."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.models = {}
        for s, g in X.groupby("stock_id"):
            self.models[s] = Hier(self.alpha, 0.0).fit(g, y.loc[g.index])
        return self

    def predict(self, X):
        out = np.full(len(X), np.nan)
        sid = X["stock_id"].to_numpy()
        for s, m in self.models.items():
            mask = sid == s
            out[mask] = m.predict(X[mask])
        return out


def _dates(X):
    return X.index.get_level_values("Date")


def select(X, y, before, h, n_folds=4):
    """Pick (alpha, c) for the hierarchical model, and alpha for the separate
    model, by purged expanding-window CV on dates < `before`."""
    d = _dates(X)
    ud = d[(d < before) & y.notna().values].unique().sort_values()
    edges = np.linspace(len(ud) * 0.5, len(ud), n_folds + 1).astype(int)
    folds = [(ud[a], ud[b - 1]) for a, b in zip(edges[:-1], edges[1:])]

    def cv(make):
        err = []
        for lo, hi in folds:
            cut = ud[max(ud.get_loc(lo) - h, 0)]
            tr, va = (d < cut) & y.notna().values, (d >= lo) & (d <= hi) & y.notna().values
            p = make().fit(X[tr], y[tr]).predict(X[va])
            err.append(np.mean((y[va].clip(-4, 4).to_numpy() - p) ** 2))
        return float(np.mean(err))

    grid = {(a, c): cv(lambda a=a, c=c: Hier(a, c)) for a in ALPHAS for c in CS}
    sep = {a: cv(lambda a=a: Separate(a)) for a in ALPHAS}
    best = min(grid, key=grid.get)
    return {"alpha": best[0], "c": best[1], "grid": grid,
            "sep_alpha": min(sep, key=sep.get), "sep_grid": sep}


def walk_forward(make, X, y_norm, scale, y_raw, test_start, h):
    d = _dates(X)
    ud = d.unique().sort_values()
    lab = y_norm.notna().values
    test = ud[ud >= test_start]
    out = []
    for k in range(0, len(test), RETRAIN_EVERY):
        block = test[k:k + RETRAIN_EVERY]
        cut = ud[max(ud.get_loc(block[0]) - h, 0)]
        tr = (d < cut) & lab
        m = make().fit(X[tr], y_norm[tr])
        te = d.isin(block)
        f = pd.DataFrame({"pred_norm": m.predict(X[te]), "scale": scale[te].to_numpy()}, index=X.index[te])
        f["mu"] = f["pred_norm"] * f["scale"]
        f["train_mean"] = y_norm[tr].mean() * f["scale"]
        out.append(f)
    res = pd.concat(out)
    res["y"] = y_raw.reindex(res.index)
    return res


def scenario_from_slopes(slopes, sd, avg_scale):
    """Linear model: predicted % change in h-day return per stock for each scenario.

    `slopes` are per 1 training-sd of each feature; ("set", a, b) shocks are
    converted to sd units with the feature's training sd."""
    rows = {}
    for key, (_, spec) in SCENARIOS.items():
        tot = pd.Series(0.0, index=slopes.index)
        for f, how in spec.items():
            if f not in slopes:
                continue
            dz = (how[2] - how[1]) / sd[f] if isinstance(how, tuple) else how
            tot += slopes[f] * dz
        rows[key] = tot * avg_scale.reindex(slopes.index) * 100
    return pd.DataFrame(rows).T


def bootstrap(X, y, alpha, c, avg_scale, n=60, block="Q", seed=0):
    """Block bootstrap over calendar quarters (keeps autocorrelation inside a
    block). Returns draws of (total response, stock-specific response). The
    stock-specific part uses only the deviation slopes d, so differences that
    come purely from each stock's volatility are not counted as stock-specific."""
    rng = np.random.default_rng(seed)
    d = _dates(X)
    q = d.to_period(block)
    uq = q.unique()
    lab = y.notna().values
    draws = []
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        idx = np.concatenate([np.flatnonzero((q == p) & lab) for p in pick])
        m = Hier(alpha, c).fit(X.iloc[idx], y.iloc[idx])
        draws.append((scenario_from_slopes(m.slopes(), m.sd, avg_scale),
                      scenario_from_slopes(m.dev, m.sd, avg_scale)))
    return draws
