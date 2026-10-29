"""HTML table: accuracy from 1 week down to 1 minute."""
import numpy as np

from report import CSS, _num, _pct

EXTRA = """th.grp{text-align:center;border-bottom:1px solid var(--axis);color:var(--ink)}
td.h{font-weight:600}.muted{color:var(--muted)}td.na{text-align:left;color:var(--muted)}"""


def _c(x):
    return "good" if x > 0 else "bad"


def _conclusions(t, cost):
    r = t.dropna(subset=["n"])
    first, last = r.iloc[0], r.iloc[-1]
    out = [f"<b>绝对误差随时间缩短而变小:</b>价格误差从{first['label']}的 {_pct(first['price_err_model'], 2)} 降到"
           f"{last['label']}的 {_pct(last['price_err_model'], 3)};但「下一刻价格=现在价格」同样从 "
           f"{_pct(first['price_err_rw'], 2)} 降到 {_pct(last['price_err_rw'], 3)}。时间越短,价格越来不及动,"
           "任何猜法的误差都会变小。"]
    sig = r[(r["hit_edge"] > 0) & (r["hit_edge_p"] < 0.05)]
    out.append("<b>方向命中率相对基准:</b>" + (
        "显著超过「总猜多数方向」的预测期:" + "、".join(f"{x['label']}(+{x['hit_edge'] * 100:.1f} 个百分点,p={x['hit_edge_p']:.3f})"
                                       for _, x in sig.iterrows()) + "。"
        if len(sig) else "没有任何一个预测期显著超过「总猜多数方向」的命中率。")
        + f"各预测期命中率在 {_pct(r['hit_rate'].min())}–{_pct(r['hit_rate'].max())} 之间。")
    ic_sig = r[r["ic_t"].abs() >= 2]
    out.append("<b>排序能力(IC):</b>" + (
        "|t|≥2 的预测期:" + "、".join(f"{x['label']}({x['ic']:+.3f},t={x['ic_t']:.1f})" for _, x in ic_sig.iterrows()) + "。"
        if len(ic_sig) else "没有任何一个预测期 |t|≥2。")
        + "短周期样本多,同样大小的 IC 更容易显著;长周期相反。")
    intr = r[r["data"].str.contains("K 线")]
    if len(intr):
        beat = intr[intr["edge_per_trade_bp"] > cost]
        out.append(f"<b>扣掉交易成本还剩多少:</b>NVDA 买卖价差约 1 美分,来回一次成本约 {cost:g}bp(还没算手续费和冲击成本)。"
                   "按预测方向每次交易、减去一直持有的收益,日内每笔的超额收益:"
                   + "、".join(f"{x['label']} {x['edge_per_trade_bp']:+.2f}bp" for _, x in intr.iterrows())
                   + ("。超过成本的只有:" + "、".join(beat["label"]) + ",但要看上面的显著性。" if len(beat)
                      else "。没有一个超过成本 — 就算有信号,也被价差吃掉了。"))
    out.append(f"<b>风险区间:</b>80% 区间实际覆盖 {_pct(r['coverage_80'].min())}–{_pct(r['coverage_80'].max())}。"
               "日内的区间按时段调整(开盘和收盘波动大、午盘小)。")
    out.append("<b>数据的限制:</b>日内 K 线只有 Yahoo 保留的最近一段(1 小时约 2 年,5–30 分钟约 60 天,1–2 分钟约 30 天),"
               "而且其中一半用来训练,所以短周期的样本外区间只有几周,碰上的市场环境很有限。"
               "缓存会在每次运行时追加新 K 线,定期重跑可以积累更长的历史。"
               "30 秒及更短需要逐笔成交/报价数据(如 Polygon、Databento 或券商接口),Yahoo 不提供。")
    return "".join(f"<li>{x}</li>" for x in out)


def build(t, cost, path):
    rows = []
    for _, r in t.iterrows():
        if np.isnan(r.get("n", np.nan)):
            rows.append(f"<tr><td class='h'>{r['label']}</td><td class='na' colspan='14'>{r['data']}</td></tr>")
            continue
        rows.append(
            f"<tr><td class='h'>{r['label']}</td><td class='muted'>{r['data']}</td><td class='muted'>{r['test_window']}</td>"
            f"<td>{int(r['n'])}{'' if r['n_eff'] == r['n'] else ' (' + str(int(r['n_eff'])) + ')'}</td>"
            f"<td>{r['typical_move_bp']:.1f}</td>"
            f"<td>{_pct(r['price_err_model'], 3)}</td><td class='muted'>{_pct(r['price_err_rw'], 3)}</td>"
            f"<td>{_pct(r['hit_rate'])}</td><td class='muted'>{_pct(r['hit_base'])}</td>"
            f"<td class='{_c(r['hit_edge'])}'>{r['hit_edge'] * 100:+.1f}</td><td>{_num(r['hit_edge_p'], 2)}</td>"
            f"<td class='{_c(r['ic'])}'>{_num(r['ic'])}</td><td>{_num(r['ic_t'], 2)}</td>"
            f"<td>{_pct(r['coverage_80'])}</td>"
            f"<td class='{_c(r['edge_per_trade_bp'] - cost)}'>{r['edge_per_trade_bp']:+.2f}</td></tr>")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>从一周到一分钟</title>
<style>{CSS}{EXTRA}</style></head><body><main>
<h1>NVDA:从一周到一分钟,预测有多准?</h1>
<p class="sub">1 周和 1 天用日线外部因子模型(154 个因子);1 小时到 1 分钟用日内 K 线模型(NVDA 自身走势、QQQ/SMH/SPY 联动、
日内 VIX、成交量、时段、开盘动量、杠杆 ETF 收盘调仓窗口、财报/期权到期日,并把日线模型对当天的预测作为一个因子)。
两者都是 Ridge + LightGBM,滚动重训,只用当时已知的数据。</p>

<h2>准确率表</h2>
<div class="card tw"><table>
<tr><th colspan="5"></th><th class="grp" colspan="2">价格误差</th><th class="grp" colspan="4">涨跌方向</th>
<th class="grp" colspan="2">排序</th><th class="grp">风险</th><th class="grp">交易</th></tr>
<tr><th>预测期</th><th>数据</th><th>样本外区间</th><th>样本数</th><th>典型波动 bp</th>
<th>模型</th><th>不预测</th><th>命中率</th><th>多数方向</th><th>超出(百分点)</th><th>p 值</th>
<th>IC</th><th>IC t</th><th>80%覆盖</th><th>每笔超额 bp</th></tr>
{''.join(rows)}</table></div>
<p class="note">不预测 = 直接用现在的价格当下一刻的预测。多数方向 = 样本外里涨和跌哪个多就一直猜哪个。
样本数括号里是有效样本(多日窗口互相重叠,按 样本数÷天数 计),p 值和 IC t 都按有效样本计算。
典型波动 = 该时间尺度上价格平均变动多少 bp(1bp = 0.01%)。每笔超额 = 按预测方向交易一次的平均收益减去一直持有,未扣成本;
绿色表示超过约 {cost:g}bp 的来回价差成本。</p>

<h2>结论</h2>
<div class="card"><ul class="concl">{_conclusions(t, cost)}</ul></div>
<p class="note">不构成投资建议。</p>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
