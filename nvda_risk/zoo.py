"""A wider set of candidate models for the short-horizon study.

Every entry has fit(X, y) -> self and predict(X) -> array. Rules ignore the
training data, so they are honest "no-fitting" benchmarks for known
intraday effects.
"""
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tuning

warnings.filterwarnings("ignore")


class Rule:
    """mu = sign * feature (no fitting)."""

    def __init__(self, col, sign):
        self.col, self.sign = col, sign

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.sign * X[self.col].fillna(0).to_numpy()


class Sk:
    """Any sklearn regressor behind median-impute (+ optional scaling), with the
    usual 1%/99% target clip and the >50%-observed column filter."""

    def __init__(self, make, scale=True):
        self.make, self.scale = make, scale

    def fit(self, X, y):
        lo, hi = y.quantile([0.01, 0.99])
        self.keep = X.columns[X.notna().mean() > 0.5]
        steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if self.scale else []) + [self.make()]
        self.m = make_pipeline(*steps).fit(X[self.keep], y.clip(lo, hi))
        return self

    def predict(self, X):
        return self.m.predict(X[self.keep])


class Logit(Sk):
    """Direction classifier; forecast = P(up) - 0.5 (only its sign and rank are used)."""

    def fit(self, X, y):
        self.keep = X.columns[X.notna().mean() > 0.5]
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegressionCV(Cs=8, cv=TimeSeriesSplit(4), max_iter=500)
                               ).fit(X[self.keep], (y > 0).astype(int))
        return self

    def predict(self, X):
        return self.m.predict_proba(X[self.keep])[:, 1] - 0.5


class Tuned:
    """Wraps the configs in tuning.py so they can sit in the same list."""

    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        self.f = tuning.Fitted(self.name, X, y)
        return self

    def predict(self, X):
        return self.f.predict(X)


class Average:
    def __init__(self, names):
        self.names = names

    def fit(self, X, y):
        self.ms = [make(n).fit(X, y) for n in self.names]
        return self

    def predict(self, X):
        ps = [m.predict(X) for m in self.ms]
        ps = [(p - p.mean()) / (p.std() + 1e-12) for p in ps]   # put members on one scale
        return np.mean(ps, axis=0)


_TS = TimeSeriesSplit(4)
MODELS = {
    # rules
    "rule_reversal": lambda: Rule("ret1", -1),
    "rule_momentum": lambda: Rule("ret1", +1),
    "rule_intraday_mom": lambda: Rule("intraday_mom", +1),
    "rule_qqq_lead": lambda: Rule("QQQ_ret1", +1),
    # linear
    "ridge": lambda: Sk(lambda: RidgeCV(alphas=np.logspace(1, 6, 21), cv=_TS)),
    "lasso": lambda: Sk(lambda: LassoCV(cv=_TS, alphas=30, max_iter=5000)),
    "elasticnet": lambda: Sk(lambda: ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=_TS, alphas=20, max_iter=5000)),
    "logit": lambda: Logit(None),
    # trees
    "lgbm_small": lambda: Tuned("lgbm_small"),
    "lgbm_medium": lambda: Tuned("lgbm_medium"),
    "lgbm_large": lambda: Tuned("lgbm_large"),
    "random_forest": lambda: Sk(lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=50, max_features=0.3,
                                                               n_jobs=-1, random_state=0), scale=False),
    "extra_trees": lambda: Sk(lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=50, max_features=0.3,
                                                           n_jobs=-1, random_state=0), scale=False),
    "hist_gb": lambda: Sk(lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200,
                                                                 l2_regularization=1.0, random_state=0), scale=False),
    # neural net
    "mlp": lambda: Sk(lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True, max_iter=300, random_state=0)),
    # blends
    "blend_small": lambda: Tuned("blend_small"),
    "avg_all_fitted": lambda: Average(["ridge", "lasso", "lgbm_small", "random_forest", "extra_trees", "mlp"]),
}
FAMILY = {k: ("规则" if k.startswith("rule") else "线性" if k in ("ridge", "lasso", "elasticnet", "logit")
              else "神经网络" if k == "mlp" else "组合" if k in ("blend_small", "avg_all_fitted") else "树模型")
          for k in MODELS}
LABEL = {"rule_reversal": "规则:短期反转", "rule_momentum": "规则:短期动量", "rule_intraday_mom": "规则:开盘动量→收盘",
         "rule_qqq_lead": "规则:跟随 QQQ", "ridge": "Ridge", "lasso": "Lasso", "elasticnet": "ElasticNet",
         "logit": "逻辑回归(方向)", "lgbm_small": "LightGBM 小", "lgbm_medium": "LightGBM 中",
         "lgbm_large": "LightGBM 大", "random_forest": "随机森林", "extra_trees": "极端随机树",
         "hist_gb": "HistGB", "mlp": "神经网络 MLP", "blend_small": "Ridge+LGBM 小", "avg_all_fitted": "多模型平均"}


def make(name):
    return MODELS[name]()
