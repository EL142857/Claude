"""HTML report for the pooled panel model: how each stock reacts to each factor shock."""
import json

import numpy as np
import pandas as pd

from report import CSS, _num, _pct

GROUP_ZH = {
    "own_technical": "个股自身技术面", "peers": "同业大型科技股", "stock_positioning": "个股量化仓位/微观结构",
    "events": "财报日历", "identity": "股票身份", "market": "大盘", "volatility": "波动率结构",
    "semis": "半导体板块", "supply_chain": "供应链", "macro": "宏观市场", "rates": "利率", "credit": "信用",
    "liquidity": "美联储流动性", "commodities": "大宗商品", "geopolitics": "地缘(中国/台湾)",
    "fund_flows": "量化基金整体仓位",
}

EXTRA_CSS = """
.chart.hm{height:560px}.chart.hm2{height:460px}
:root{--div-neg:#e34948;--div-mid:#f0efec;--div-pos:#2a78d6;--seq-lo:#cde2fb;--seq-hi:#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--div-neg:#e66767;--div-mid:#383835;
--div-pos:#3987e5;--seq-lo:#1a1a19;--seq-hi:#86b6ef}}
:root[data-theme="dark"]{--div-neg:#e66767;--div-mid:#383835;--div-pos:#3987e5;--seq-lo:#1a1a19;--seq-hi:#86b6ef}
"""

JS = """
const D = __DATA__;
function v(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
const cfg={displayModeBar:false,responsive:true};
function lay(extra){return Object.assign({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:v('--ink2'),size:12},margin:{l:200,r:16,t:40,b:16},
  xaxis:{side:'top',gridcolor:'rgba(0,0,0,0)'},yaxis:{autorange:'reversed',gridcolor:'rgba(0,0,0,0)'}},extra||{})}
function heat(id,M,div,fmt,hover){
  const flat=M.z.flat().filter(x=>x!==null), m=Math.max(...flat.map(Math.abs));
  const cs=div?[[0,v('--div-neg')],[0.5,v('--div-mid')],[1,v('--div-pos')]]:[[0,v('--seq-lo')],[1,v('--seq-hi')]];
  Plotly.react(id,[{type:'heatmap',z:M.z,x:M.x,y:M.y,colorscale:cs,zmin:div?-m:0,zmax:m,
    xgap:2,ygap:2,text:M.text,texttemplate:'%{text}',textfont:{color:v('--ink'),size:11},
    customdata:M.hover,hovertemplate:hover,colorbar:{thickness:10,outlinewidth:0,tickfont:{color:v('--muted')}}}],
    lay(),cfg);
}
function render(){
  heat('h_model',D.model,true,'','%{y} → %{x}<br>模型预测 5 日收益变化 %{z:.2f}%<br>%{customdata}<extra></extra>');
  heat('h_emp',D.emp,true,'','%{y} → %{x}<br>历史实测 %{z:.2f}% / 1σ<br>%{customdata}<extra></extra>');
  heat('h_shap',D.shap,false,'','%{x} · %{y}<br>占该股预测驱动的 %{z:.1f}%<extra></extra>');
  const E=D.eq, c=[v('--s1'),v('--s2'),v('--s3')];
  Plotly.react('c_eq',E.series.map((s,i)=>({x:E.d,y:s.y,name:s.name,line:{color:c[i],width:2},
    hovertemplate:'%{y:.2f}'})),{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:v('--ink2'),size:12},margin:{l:48,r:16,t:16,b:40},hovermode:'x unified',
    legend:{orientation:'h',y:1.1,x:0},xaxis:{gridcolor:v('--grid'),linecolor:v('--axis')},
    yaxis:{title:'净值(起点=1)',gridcolor:v('--grid'),linecolor:v('--axis')}},cfg);
}
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
"""


def _matrix(df, fmt, hover=None):
    z = df.values.round(4)
    return {"x": list(df.columns), "y": list(df.index),
            "z": [[None if np.isnan(c) else float(c) for c in r] for r in z],
            "text": [[("" if np.isnan(c) else fmt.format(c)) for c in r] for r in z],
            "hover": hover if hover is not None else [["" for _ in r] for r in z]}


def _conclusions(res, resp, emp_b, emp_t, shares, labels):
    out = []
    ps = res["per_stock"]
    good = [t for t, m in ps.items() if m["ic_spearman"] > 0.03]
    beat = [t for t, m in ps.items() if m["hit_rate"] > m["hit_rate_always_up"] + 0.01]
    out.append(f"<b>预测力:</b>同一个模型在 {len(ps)} 只股票上,IC>0.03 的有 {len(good)} 只"
               f"({'、'.join(good) or '无'});方向命中率超过「永远看涨」的有 {len(beat)} 只"
               f"({'、'.join(beat) or '无'})。区间覆盖率在 "
               f"{min(m['coverage_80'] for m in ps.values()) * 100:.0f}%–{max(m['coverage_80'] for m in ps.values()) * 100:.0f}%"
               "(目标 80%),风险刻画依然是最稳的部分。")
    xs = res["cross_section"]
    out.append(f"<b>股票之间的相对排序:</b>每日截面 rank IC 均值 {xs['rank_ic_mean']:+.3f},"
               f"非重叠 t 值 {xs['rank_ic_tstat_nonoverlap']:.2f}"
               + ("(统计显著)。" if abs(xs["rank_ic_tstat_nonoverlap"]) >= 2 else "(不显著)。")
               + f"多前 3 空后 3 组合夏普 {xs['perf']['long_short']['sharpe']:.2f}。")
    if res.get("nvda_single_model_ic") is not None:
        out.append(f"<b>NVDA:合并模型 vs 单独模型:</b>样本外 5 日 IC "
                   f"{ps['NVDA']['ic_spearman']:+.3f} vs 单独 NVDA 模型 {res['nvda_single_model_ic']:+.3f}。")
    # widest spread of reactions across stocks
    spread = (resp.max(axis=1) - resp.min(axis=1)).sort_values(ascending=False)
    lines = []
    for k in spread.index[:5]:
        r = resp.loc[k]
        lines.append(f"{labels[k]}:{r.idxmin()} {r.min():+.2f}% … {r.idxmax()} {r.max():+.2f}%")
    out.append("<b>各股反应差异最大的冲击:</b>" + ";".join(lines) + "。")
    sig = (emp_t.abs() >= 2).sum().sum()
    out.append(f"<b>历史实测核对:</b>{emp_t.size} 个(冲击×股票)组合里,历史回归 |t|≥2 的有 {sig} 个"
               f"(纯随机约 {emp_t.size * 0.05:.0f} 个)。模型反应与历史实测的整体一致性 "
               f"{res['agreement']['model_vs_history_all']:+.2f},去掉每个冲击的平均反应后只看股票间差异 "
               f"{res['agreement']['model_vs_history_pattern']:+.2f};和只用 2024 年前数据训练的同一模型相比 "
               f"{res['agreement']['model_vs_pre2024_all']:+.2f}。")
    top_q = shares.get("fund_flows")
    if top_q is not None:
        out.append(f"<b>量化基金整体仓位的影响力:</b>占各股预测驱动的 {top_q.min() * 100:.0f}%–{top_q.max() * 100:.0f}%,"
                   f"最受影响的是 {top_q.idxmax()},最不受影响的是 {top_q.idxmin()}。")
    return "".join(f"<li>{x}</li>" for x in out)


def build(res, resp, resp_early, emp_b, emp_t, shares, pnls, path):
    labels = res["scenario_labels"]
    cols = res["universe"]
    resp = resp[cols]
    emp_b, emp_t = emp_b[cols], emp_t[cols]
    hover_m = [[f"2024 年前训练的同一模型:{resp_early.loc[k, c]:+.2f}% · 历史实测 {emp_b.loc[k, c]:+.2f}% (t={emp_t.loc[k, c]:.1f})"
                for c in cols] for k in resp.index]
    hover_e = [[f"Newey-West t = {emp_t.loc[k, c]:.2f}" for c in cols] for k in emp_b.index]
    rz = resp.rename(index=labels)
    ez = emp_b.rename(index=labels)
    sh = shares.T[cols] * 100                                   # groups x stocks, in %
    sh = sh.rename(index=GROUP_ZH).sort_values(cols[0], ascending=False)
    eq = (1 + pd.DataFrame(pnls).fillna(0)).cumprod()
    names = {"long_short": "多前3 / 空后3", "long_top3": "只做多前3", "equal_weight": "9 只等权"}
    payload = {
        "model": _matrix(rz, "{:+.2f}", hover_m),
        "emp": _matrix(ez, "{:+.2f}", hover_e),
        "shap": _matrix(sh, "{:.0f}"),
        "eq": {"d": [str(x.date()) for x in eq.index],
               "series": [{"name": names[c], "y": eq[c].round(4).tolist()} for c in eq]},
    }
    rows = "".join(
        f"<tr><td>{t}</td><td>{m['n']}</td><td>{_num(m['ic_spearman'])}</td><td>{_num(m['monthly_ic_tstat'], 2)}</td>"
        f"<td>{_pct(m['hit_rate'])}</td><td>{_pct(m['hit_rate_always_up'])}</td><td>{_pct(m['share_pred_up'], 0)}</td>"
        f"<td>{_pct(m['oos_r2_vs_drift'], 2)}</td><td>{_pct(m['coverage_80'])}</td><td>{_pct(m['coverage_95'])}</td></tr>"
        for t, m in res["per_stock"].items())
    xs = res["cross_section"]
    prow = "".join(
        f"<tr><td>{names[k]}</td><td>{_pct(p['total_return'])}</td><td>{_pct(p['cagr'])}</td>"
        f"<td>{_num(p['sharpe'], 2)}</td><td>{_pct(p['max_drawdown'])}</td></tr>" for k, p in xs["perf"].items())

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>科技股因子反应</title>
<style>{CSS}{EXTRA_CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script></head><body><main>
<h1>一个模型,九只科技股:同一量化冲击下谁反应最大</h1>
<p class="sub">{'、'.join(cols)} · 合并训练一个 LightGBM({res['n_features']} 个因子,{res['n_rows']:,} 行)·
预测 {res['horizon']} 日收益 · 样本外 {res['test_start']} 至 {res['last_date']}</p>

<h2>结论</h2>
<div class="card"><ul class="concl">{_conclusions(res, resp, emp_b, emp_t, shares, labels)}</ul></div>

<h2>冲击反应图:模型怎么看</h2>
<p class="note">把一组因子同时推高(或压低)1 个标准差,其他不变,看模型对每只股票未来 {res['horizon']} 日收益的预测变化多少(%)。
在 2024 年以来的每一天都做一遍再取平均。蓝色 = 预测更涨,红色 = 预测更跌。悬停看 2024 年前训练版本和历史实测。</p>
<div class="card tw"><div id="h_model" class="chart hm"></div></div>

<h2>冲击反应图:历史实际怎么样</h2>
<p class="note">不经过模型,直接用 2014 年以来的数据:每只股票未来 {res['horizon']} 日收益对该冲击的综合分数做回归,
单位是"冲击分数每增加 1σ,收益变化 %"(期权到期周、财报是"有 vs 没有"的差)。悬停看 Newey-West t 值,|t|<2 的格子多半是噪声。
和上图的单位不完全一样,应该比较的是颜色格局。</p>
<div class="card tw"><div id="h_emp" class="chart hm"></div></div>

<h2>每只股票的预测由哪些因子驱动</h2>
<p class="note">SHAP 贡献:同一个模型,对不同股票,各组因子各占其预测驱动力的百分比(每列加起来 100%)。</p>
<div class="card tw"><div id="h_shap" class="chart hm2"></div></div>

<h2>同一个模型在每只股票上的样本外表现</h2>
<div class="card tw"><table><tr><th>股票</th><th>样本数</th><th>IC</th><th>月度IC t</th><th>命中率</th>
<th>永远看涨命中率</th><th>预测看涨占比</th><th>R² vs 漂移</th><th>80%覆盖</th><th>95%覆盖</th></tr>{rows}</table></div>

<h2>用模型给九只股票排序</h2>
<p class="note">每天按预测收益排序,多前 3 只、空后 3 只(或只做多前 3),次日调仓,每单位换手扣 5bp。
截面 rank IC 均值 {xs['rank_ic_mean']:+.3f},每 {res['horizon']} 天取一次的非重叠 t 值 {xs['rank_ic_tstat_nonoverlap']:.2f}。</p>
<div class="card"><div id="c_eq" class="chart"></div></div>
<div class="card tw"><table><tr><th>组合</th><th>总收益</th><th>年化</th><th>夏普</th><th>最大回撤</th></tr>{prow}</table></div>

<h2>方法</h2>
<p class="note">每行是(日期,股票)。大盘、波动率、利率、信用、流动性、大宗、地缘、半导体板块、供应链和量化基金整体仓位
(纳指 CTA、波动率目标、风险平价、动量拥挤、期权到期/月末日历)对所有股票相同;个股技术面、同业篮子、个股 CTA 仓位、
相对纳指的统计套利残差、对 SMH/动量因子的 beta、微观结构和财报日历按股票分别计算。目标是 {res['horizon']} 日收益除以该股自身的事前波动,
避免 TSLA 这类高波动股票主导训练;模型通过 stock_id 和个股特征学到"不同股票对同一因子反应不同"。
每 21 个交易日滚动重训,剔除与测试期重叠的标签。不构成投资建议。</p>
</main><script>{JS.replace('__DATA__', json.dumps(payload))}</script></body></html>"""
    path.write_text(html, encoding="utf-8")
