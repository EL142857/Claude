"""HTML page: forecast accuracy by horizon."""
import json

from report import CSS, _num, _pct

EXTRA_CSS = """
th.grp{text-align:center;border-bottom:1px solid var(--axis);color:var(--ink)}
td.h{font-weight:600}.muted{color:var(--muted)}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:4px 0 0 4px}
.sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
"""

JS = """
const D=__DATA__;
function v(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function render(){
  const x=D.h.map(h=>h+'日');
  Plotly.react('c_err',[
    {x,y:D.rw,name:'随机游走(未来价=今天价)',mode:'lines+markers',line:{color:v('--s2'),width:2},marker:{size:8},
     hovertemplate:'%{y:.2f}%'},
    {x,y:D.model,name:'模型',mode:'lines+markers',line:{color:v('--s1'),width:2},marker:{size:8},
     hovertemplate:'%{y:.2f}%'}],
    {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:v('--ink2'),size:12},
     margin:{l:48,r:16,t:16,b:40},hovermode:'x unified',showlegend:false,
     xaxis:{title:'预测期(交易日,从长到短)',gridcolor:'rgba(0,0,0,0)',linecolor:v('--axis')},
     yaxis:{title:'平均价格误差 %',gridcolor:v('--grid'),linecolor:v('--axis'),rangemode:'tozero'}},
    {displayModeBar:false,responsive:true});
}
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
"""


def _cls(x, good=0.0):
    return "good" if x > good else "bad"


def _conclusions(t):
    hs = list(t.index)
    out = [f"<b>绝对准确度随预测期缩短而提高:</b>平均价格误差从 {hs[0]} 日的 {_pct(t.loc[hs[0], 'price_err_model'])} "
           f"降到 1 日的 {_pct(t.loc[1, 'price_err_model'])}。但「未来价=今天价」这个零信息的猜法也同样从 "
           f"{_pct(t.loc[hs[0], 'price_err_rw'])} 降到 {_pct(t.loc[1, 'price_err_rw'])} — 误差变小主要是因为时间短、价格来不及走远,不是模型变聪明了。"]
    beat = [h for h in hs if t.loc[h, "price_err_gain"] > 0.005]
    out.append("<b>价格误差相对随机游走:</b>" + (
        f"模型在 {'、'.join(f'{h} 日' for h in beat)} 比随机游走误差小 0.5% 以上(相对);"
        if beat else "没有任何一个预测期比随机游走的价格误差小 0.5% 以上(相对);")
        + "其余预测期两者几乎一样。")
    edge = [h for h in hs if t.loc[h, "hit_edge"] > 0.01]
    sig = [h for h in hs if t.loc[h, "hit_edge_p"] < 0.05 and t.loc[h, "hit_edge"] > 0]
    out.append(f"<b>方向:</b>命中率在长预测期看起来更高(最高 {_pct(t['hit_rate'].max())}),但那是因为 NVDA 这两年大部分时间在涨,"
               f"「永远看涨」在 {hs[0]} 日能拿到 {_pct(t.loc[hs[0], 'hit_always_up'])}。超过这个基准 1 个百分点以上的预测期:"
               f"{'、'.join(f'{h} 日' for h in edge) or '无'};按有效样本数检验显著(p<0.05)的:{'、'.join(f'{h} 日' for h in sig) or '无'}。")
    ic_sig = [h for h in hs if abs(t.loc[h, "ic_t_eff"]) >= 2]
    ic_txt = "、".join(f"{h} 日({t.loc[h, 'ic']:+.3f})" for h in ic_sig) or "无"
    out.append(f"<b>排序能力(IC):</b>各预测期 IC 在 {t['ic'].min():+.3f} 到 {t['ic'].max():+.3f} 之间;"
               f"按有效样本 |t|≥2 的:{ic_txt}。")
    out.append(f"<b>风险区间在每个预测期都校准良好:</b>80% 区间实际覆盖 {_pct(t['coverage_80'].min())}–{_pct(t['coverage_80'].max())},"
               f"区间宽度从 {hs[0]} 日的 ±{t.loc[hs[0], 'band80_width'] * 50:.1f}% 收窄到 1 日的 ±{t.loc[1, 'band80_width'] * 50:.1f}%。"
               "这是模型唯一在所有时间尺度上都可靠的输出。")
    out.append(f"<b>长预测期的数字要打折:</b>样本外只有约 {int(t.loc[1, 'n'])} 个交易日,{hs[0]} 日预测互不重叠的样本只有约 "
               f"{int(t.loc[hs[0], 'n_eff'])} 个,所以长期限的命中率和 IC 波动很大。")
    return "".join(f"<li>{x}</li>" for x in out)


def build(t, test_start, last_date, path):
    rows = []
    for h, r in t.iterrows():
        rows.append(
            f"<tr><td class='h'>{h} 日</td>"
            f"<td>{_pct(r['price_err_model'], 2)}</td><td class='muted'>{_pct(r['price_err_rw'], 2)}</td>"
            f"<td class='{_cls(r['price_err_gain'])}'>{r['price_err_gain'] * 100:+.2f}%</td>"
            f"<td>{_pct(r['hit_rate'])}</td><td class='muted'>{_pct(r['hit_always_up'])}</td>"
            f"<td class='{_cls(r['hit_edge'])}'>{r['hit_edge'] * 100:+.1f}</td><td>{_num(r['hit_edge_p'], 2)}</td>"
            f"<td class='{_cls(r['ic'])}'>{_num(r['ic'])}</td><td>{_num(r['ic_t_eff'], 2)}</td>"
            f"<td class='{_cls(r['r2_vs_drift'])}'>{_pct(r['r2_vs_drift'], 2)}</td>"
            f"<td>{_pct(r['coverage_80'])}</td><td>{_pct(r['coverage_95'])}</td>"
            f"<td>±{r['band80_width'] * 50:.1f}%</td><td class='muted'>{int(r['n_eff'])}</td></tr>")
    payload = {"h": [int(h) for h in t.index],
               "model": [round(x * 100, 3) for x in t["price_err_model"]],
               "rw": [round(x * 100, 3) for x in t["price_err_rw"]]}
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>预测期准确率表</title>
<style>{CSS}{EXTRA_CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script></head><body><main>
<h1>NVDA:预测期越短,预测越准吗?</h1>
<p class="sub">同一套 154 个因子和同一个模型(Ridge + LightGBM),分别预测未来 60 到 1 个交易日 · 样本外 {test_start} 至 {last_date} ·
每 21 个交易日滚动重训</p>

<h2>准确率表</h2>
<div class="card tw"><table>
<tr><th></th><th class="grp" colspan="3">价格误差(越小越好)</th><th class="grp" colspan="4">涨跌方向</th>
<th class="grp" colspan="3">排序与拟合</th><th class="grp" colspan="3">风险区间</th><th></th></tr>
<tr><th>预测期</th><th>模型</th><th>随机游走</th><th>模型改善</th>
<th>命中率</th><th>永远看涨</th><th>超出(百分点)</th><th>p 值</th>
<th>IC</th><th>IC t</th><th>R² vs 漂移</th>
<th>80%覆盖</th><th>95%覆盖</th><th>80%区间</th><th>有效样本</th></tr>
{''.join(rows)}</table></div>
<p class="note">价格误差 = |预测价 − 实际价| / 实际价 的平均。随机游走 = 直接用今天的价格当预测。「永远看涨」= 每次都猜涨的命中率,方向预测至少要超过它。
p 值和 IC t 按有效样本(样本数 ÷ 预测期)计算,因为重叠的多日收益不是独立观测。R² vs 漂移 > 0 才说明模型比"按历史平均涨幅外推"更好。
80% 区间 = 模型给出的 80% 价格区间的平均半宽。</p>

<h2>结论</h2>
<div class="card"><ul class="concl">{_conclusions(t)}</ul></div>

<h2>价格误差:模型 vs 随机游走</h2>
<div class="legend"><span><span class="sw" style="background:var(--s1)"></span>模型</span>
<span><span class="sw" style="background:var(--s2)"></span>随机游走(未来价=今天价)</span></div>
<div class="card"><div id="c_err" class="chart"></div></div>
<p class="note">{"两条线几乎重合:时间越短误差越小,但模型和「什么都不做」的差距始终很小。" if t["price_err_gain"].abs().max() < 0.02
 else "时间越短误差越小;两条线的差距就是模型带来的改善。"}不构成投资建议。</p>
</main><script>{JS.replace('__DATA__', json.dumps(payload))}</script></body></html>"""
    path.write_text(html, encoding="utf-8")
