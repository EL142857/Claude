"""Is the 30-minute NVDA signal real? Wider data, many models, honest tests.

Setups (all forecast the next 30 minutes, same test half of the data):
  A  NVDA, 30m bars                       (the original finding)
  B  NVDA, 5m bars, target 6 bars ahead   (finer features, a forecast every 5 minutes)
  C  9 tech stocks pooled, 30m bars       (does it generalise beyond NVDA?)

For every setup, all models in zoo.MODELS are run walk-forward. The honest
number is the model picked by CV on the training half only. A day-block
bootstrap "reality check" asks whether the best of all models could be luck.

  python study30.py                 -> output/study30.{json,html}
  python study30.py --report-only   rebuild the page from saved results
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import intraday
import zoo
from config import UNIVERSE

OUT = Path(__file__).parent / "output"
N_CHUNKS, N_BOOT = 10, 1000
COST_BP = 0.5


def _setup_single(interval, ahead, daily_mu):
    X, y, c, _ = intraday.build(interval, daily_mu=daily_mu, refresh=False, ahead=ahead)
    ok = y.notna()
    X, y = X[ok], y[ok]
    return {"X": X.reset_index(drop=True), "y": y.reset_index(drop=True), "y_raw": y.reset_index(drop=True),
            "scale": pd.Series(1.0, index=range(len(y))), "day": pd.Series(X.index.normalize()),
            "ts": pd.Series(X.index), "ticker": pd.Series(["NVDA"] * len(y)), "ahead": ahead}


def _setup_panel(interval):
    parts = []
    for i, t in enumerate(UNIVERSE):
        X, y, c, _ = intraday.build(interval, refresh=True, target=t)
        X["stock_id"] = float(i)
        scale = X["ret1"].rolling(60, min_periods=20).std()     # stock's own recent bar vol, known at t
        ok = y.notna() & scale.notna()
        parts.append((X[ok], y[ok], scale[ok], t))
    X = pd.concat([p[0] for p in parts])
    y_raw = pd.concat([p[1] for p in parts])
    scale = pd.concat([p[2] for p in parts])
    ticker = np.concatenate([[p[3]] * len(p[1]) for p in parts])
    ts = X.index
    return {"X": X.reset_index(drop=True), "y": (y_raw / scale).reset_index(drop=True),
            "y_raw": y_raw.reset_index(drop=True), "scale": scale.reset_index(drop=True),
            "day": pd.Series(ts.normalize()), "ts": pd.Series(ts), "ticker": pd.Series(ticker), "ahead": 1}


def walk_forward(S, name):
    X, y, day = S["X"], S["y"], S["day"]
    ud = np.sort(day.unique())
    test_days = ud[len(ud) // 2:]
    pred = np.full(len(X), np.nan)
    for ch in np.array_split(test_days, N_CHUNKS):
        tr = (day < ch[0]).values
        te = day.isin(ch).values
        pred[te] = zoo.make(name).fit(X[tr], y[tr]).predict(X[te])
    return pred * S["scale"].values                               # back to return units


def cv_select(S, n_folds=3):
    """Pick a model by mean fold IC on the training half only."""
    X, y, day = S["X"], S["y"], S["day"]
    ud = np.sort(day.unique())
    train_days = ud[:len(ud) // 2]
    edges = np.linspace(len(train_days) * 0.4, len(train_days), n_folds + 1).astype(int)
    score = {}
    for name in zoo.MODELS:
        ics = []
        for a, b in zip(edges[:-1], edges[1:]):
            tr = (day < train_days[max(a - 1, 0)]).values       # one-day purge
            va = day.between(train_days[a], train_days[b - 1]).values
            p = zoo.make(name).fit(X[tr], y[tr]).predict(X[va])
            ics.append(stats.spearmanr(p, y[va]).statistic)
        score[name] = float(np.nanmean(ics))
    return max(score, key=score.get), score


def metrics(mu, S):
    m = ~np.isnan(mu)
    y = S["y_raw"].values[m]
    mu = mu[m]
    n = len(y)
    n_eff = max(n // S["ahead"], 3)
    nz = y != 0
    act = nz & (mu != 0)                      # a zero forecast is "no trade", not a miss
    hit = float((np.sign(mu[act]) == np.sign(y[act])).mean())
    up = float((y[nz] > 0).mean())
    base = max(up, 1 - up)
    ic = float(stats.spearmanr(mu, y).statistic)
    n_act = max(int(act.sum()) // S["ahead"], 3)
    out = {"n": n, "n_eff": n_eff, "n_trades": int(act.sum()), "hit": hit, "base": base, "hit_edge": hit - base,
           "hit_p": float(stats.binomtest(round(hit * n_act), n_act, base, alternative="greater").pvalue),
           "ic": ic, "ic_t": float(ic * np.sqrt(n_eff - 2) / np.sqrt(max(1 - ic ** 2, 1e-9))),
           "edge_bp": float(((np.sign(mu) * y).mean() - y.mean()) * 1e4)}
    if S["ticker"].nunique() > 1:                                  # cross-sectional rank IC per timestamp
        df = pd.DataFrame({"ts": S["ts"].values[m], "mu": mu, "y": y})
        xs = df.groupby("ts").apply(lambda g: stats.spearmanr(g["mu"], g["y"]).statistic if len(g) >= 5 else np.nan)
        xs = xs.dropna()
        out["xs_ic"] = float(xs.mean())
        out["xs_ic_t"] = float(xs.mean() / xs.std() * np.sqrt(len(xs)))
    return out


def reality_check(preds, S, seed=0):
    """Day-block bootstrap of the best model's hit edge and IC versus luck
    (White 2000, recentred). Returns p-values for 'best of all models'."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    m = ~np.isnan(preds[names[0]])
    y = S["y_raw"].values[m]
    day = S["day"].values[m]
    nz = y != 0
    hits = np.stack([(np.sign(preds[k][m]) == np.sign(y)).astype(float) for k in names])
    act = np.stack([nz & (preds[k][m] != 0) for k in names])          # traded rows per model
    up = (y[nz] > 0).mean()
    base = max(up, 1 - up)
    obs_hit = (hits * act).sum(axis=1) / act.sum(axis=1) - base
    obs_ic = np.array([stats.spearmanr(preds[k][m], y).statistic for k in names])
    ud = np.unique(day)
    rows_by_day = {d: np.flatnonzero(day == d) for d in ud}
    mx_hit, mx_ic, each_ic = [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows_by_day[d] for d in rng.choice(ud, len(ud))])
        ab = act[:, idx]
        h = (hits[:, idx] * ab).sum(axis=1) / ab.sum(axis=1) - base
        mx_hit.append(np.max(h - obs_hit))
        icb = np.array([stats.spearmanr(preds[k][m][idx], y[idx]).statistic for k in names])
        mx_ic.append(np.max(icb - obs_ic))
        each_ic.append(icb - obs_ic)
    return {"best_hit_model": names[int(np.argmax(obs_hit))], "best_hit_edge": float(obs_hit.max()),
            "rc_p_hit": float(np.mean(np.array(mx_hit) >= obs_hit.max())),
            "best_ic_model": names[int(np.argmax(obs_ic))], "best_ic": float(obs_ic.max()),
            "rc_p_ic": float(np.mean(np.array(mx_ic) >= obs_ic.max())),
            # each model on its own, day-clustered: P(recentred bootstrap IC >= observed IC)
            "single_p_ic": {k: float(np.mean(np.array(each_ic)[:, i] >= obs_ic[i])) for i, k in enumerate(names)}}


def rule_stability(name="rule_reversal", interval="30m"):
    """A rule has no training, so every day is out of sample: IC and edge in each
    half of the data and in every week, for NVDA and each stock in the pool."""
    out = {}
    for t in UNIVERSE:
        X, y, c, _ = intraday.build(interval, refresh=False, target=t)
        ok = y.notna()
        X, y = X[ok], y[ok]
        mu = zoo.make(name).predict(X)
        d = X.index.normalize()
        ud = np.sort(d.unique())
        halves = {}
        for lab, sel in (("first", d < ud[len(ud) // 2]), ("second", d >= ud[len(ud) // 2])):
            sel = np.asarray(sel)
            act = sel & (mu != 0) & (y.values != 0)
            halves[lab] = {"ic": float(stats.spearmanr(mu[sel], y[sel]).statistic),
                           "hit": float((np.sign(mu[act]) == np.sign(y.values[act])).mean()),
                           "edge_bp": float(((np.sign(mu[sel]) * y.values[sel]).mean() - y.values[sel].mean()) * 1e4)}
        wk = X.index.to_period("W")
        weekly = [float(stats.spearmanr(mu[np.asarray(wk == w)], y[np.asarray(wk == w)]).statistic)
                  for w in wk.unique() if (wk == w).sum() >= 20]
        out[t] = {"halves": halves, "weekly_ic": weekly}
    return out


def main():
    daily_mu = pd.read_csv(OUT / "tuned_fc_daily_h1.csv", index_col=0, parse_dates=True)["mu"]
    setups = {
        "A": ("NVDA · 30 分钟 K 线", lambda: _setup_single("30m", 1, daily_mu)),
        "B": ("NVDA · 5 分钟 K 线,预测未来 30 分钟", lambda: _setup_single("5m", 6, daily_mu)),
        "C": ("9 只科技股合并 · 30 分钟 K 线", lambda: _setup_panel("30m")),
    }
    res = {}
    for key, (title, make_setup) in setups.items():
        S = make_setup()
        days = np.sort(S["day"].unique())
        print(f"{key}: {title}: {len(S['X'])} rows, {S['X'].shape[1]} features, {len(days)} days")
        pick, cv = cv_select(S)
        preds, table = {}, {}
        for name in zoo.MODELS:
            preds[name] = walk_forward(S, name)
            table[name] = metrics(preds[name], S)
            print(f"   {name:16s} IC {table[name]['ic']:+.3f} (t {table[name]['ic_t']:+.2f})  "
                  f"hit edge {table[name]['hit_edge'] * 100:+.1f}pp (p {table[name]['hit_p']:.3f})  cv {cv[name]:+.3f}")
        rc = reality_check(preds, S)
        pd.DataFrame({"ts": S["ts"], "ticker": S["ticker"], "y": S["y_raw"], **preds}).to_csv(
            OUT / f"study30_preds_{key}.csv", index=False)
        per_stock = {}
        if S["ticker"].nunique() > 1:
            for t in UNIVERSE:
                mask = (S["ticker"] == t).values
                sub = {k: v[mask] if isinstance(v, pd.Series) else v for k, v in S.items()}
                sub = {k: (v.reset_index(drop=True) if isinstance(v, pd.Series) else v) for k, v in sub.items()}
                per_stock[t] = {name: metrics(preds[name][mask], sub)
                                for name in dict.fromkeys((pick, "lgbm_small", "rule_reversal"))}
        test_days = days[len(days) // 2:]
        res[key] = {"title": title, "rows": len(S["X"]), "n_features": int(S["X"].shape[1]),
                    "days": len(days), "test_window": f"{pd.Timestamp(test_days[0]):%Y-%m-%d} 至 {pd.Timestamp(test_days[-1]):%Y-%m-%d}",
                    "cv_pick": pick, "cv_scores": cv, "models": table, "reality_check": rc, "per_stock": per_stock}
        print(f"   CV pick {pick}: IC {table[pick]['ic']:+.3f}, hit edge {table[pick]['hit_edge'] * 100:+.1f}pp; "
              f"reality check p(hit)={rc['rc_p_hit']:.3f} p(IC)={rc['rc_p_ic']:.3f}")
    res["_stability"] = rule_stability()
    (OUT / "study30.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False))
    import study30_report
    study30_report.build(res, COST_BP, OUT / "study30.html")
    print("wrote", OUT / "study30.html")


if __name__ == "__main__":
    import sys
    if "--report-only" in sys.argv:                 # rebuild the page from saved results
        import study30_report
        res = json.loads((OUT / "study30.json").read_text())
        res["_stability"] = rule_stability()
        (OUT / "study30.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False))
        study30_report.build(res, COST_BP, OUT / "study30.html")
    else:
        main()
