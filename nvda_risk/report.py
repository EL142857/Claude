"""Self-contained HTML report (Chinese) with interactive Plotly charts."""
import json

import numpy as np
import pandas as pd

GROUP_ZH = {
    "own_technical": "NVDA自身技术面", "events": "财报日历", "market": "大盘",
    "volatility": "波动率结构", "semis": "半导体同业", "supply_chain": "供应链",
    "ai_demand": "AI需求方(云厂商)", "macro": "宏观市场", "rates": "利率",
    "credit": "信用", "liquidity": "美联储流动性", "commodities": "大宗商品",
    "geopolitics": "地缘(中国/台湾)", "quant_flows": "量化机构行为",
}

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300;--bad:#d03b3b;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c;--bad:#e66767;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--good:#0ca30c;--bad:#e66767;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 8px}
.sub{color:var(--ink2);margin:0 0 24px}.note{color:var(--ink2);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
.tile .d{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;margin-top:8px}
.chart{width:100%;height:380px}.small{height:220px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}
.good{color:var(--good)}.bad{color:var(--bad)}
"""

JS = """
const D = __DATA__;
function v(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function base(ytitle, extra){
  return Object.assign({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:v('--ink2'),size:12},margin:{l:56,r:16,t:16,b:40},hovermode:'x unified',
    legend:{orientation:'h',y:1.1,x:0},
    xaxis:{gridcolor:v('--grid'),linecolor:v('--axis'),zeroline:false},
    yaxis:{title:ytitle,gridcolor:v('--grid'),linecolor:v('--axis'),zeroline:false}}, extra||{});
}
const cfg={displayModeBar:false,responsive:true};
function render(){
  const s1=v('--s1'),s2=v('--s2'),s3=v('--s3');
  const P=D.price;
  Plotly.react('c_price',[
    {x:P.d,y:P.hi,line:{width:0},hoverinfo:'skip',showlegend:false,type:'scatter'},
    {x:P.d,y:P.lo,fill:'tonexty',fillcolor:s2+'26',line:{width:0},name:'80% 预测区间',type:'scatter',
     hovertemplate:'%{y:.2f}'},
    {x:P.d,y:P.pred,name:'模型预测价('+D.h+'日前做出)',line:{color:s2,width:2},hovertemplate:'%{y:.2f}'},
    {x:P.d,y:P.actual,name:'实际收盘价',line:{color:s1,width:2},hovertemplate:'%{y:.2f}'},
  ],base('美元'),cfg);
  const E=D.equity;
  Plotly.react('c_eq',[
    {x:E.d,y:E.model,name:'全因子 + 风控',line:{color:s1,width:2},hovertemplate:'%{y:.2f}'},
    {x:E.d,y:E.bh,name:'买入持有 NVDA',line:{color:s2,width:2},hovertemplate:'%{y:.2f}'},
    {x:E.d,y:E.tech,name:'仅技术面 + 风控',line:{color:s3,width:2},hovertemplate:'%{y:.2f}'},
  ],base('净值(起点=1)'),cfg);
  if(D.ablation.length){
    const A=D.ablation;
    Plotly.react('c_abl',[{type:'bar',orientation:'h',y:A.map(a=>a.name),x:A.map(a=>a.ic_drop),
      marker:{color:A.map(a=>a.ic_drop>=0?s1:v('--muted'))},
      hovertemplate:'%{y}: 去掉后 IC 变化 %{x:.4f}<extra></extra>'}],
      base('',{hovermode:'closest',margin:{l:140,r:16,t:8,b:40},
        xaxis:{title:'去掉该组后 IC 下降(越大越重要)',gridcolor:v('--grid'),zeroline:true,zerolinecolor:v('--axis')},
        yaxis:{autorange:'reversed',gridcolor:'rgba(0,0,0,0)'}}),cfg);
  }
  D.quant.forEach((q,i)=>Plotly.react('c_q'+i,[{x:q.d,y:q.y,line:{color:s1,width:2},
    name:q.name,hovertemplate:'%{y:.2f}<extra></extra>'}],base('',{margin:{l:40,r:8,t:8,b:30}}),cfg));
}
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
"""


def _pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _num(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _cls(x):
    return "good" if x > 0 else "bad"


def build(results, price, fc, pnl, pnl_base, bh, quant, path):
    h = results["latest_forecast"]["horizon"]
    m = results["horizons"][h]["full_model"]
    st = results["strategy"]
    lf = results["latest_forecast"]

    # predicted price made at t, plotted at its target date t+h
    idx = price.index
    pos = idx.get_indexer(fc.index)
    tgt = [idx[p + h] if p + h < len(idx) else None for p in pos]
    keep = [t is not None for t in tgt]
    f = fc[keep]
    tdates = [t for t in tgt if t is not None]
    p0 = price.reindex(f.index).values
    pdata = {
        "d": [str(t.date()) for t in tdates],
        "actual": np.round(price.reindex(tdates).values, 2).tolist(),
        "pred": np.round(p0 * np.exp(f["mu"].values), 2).tolist(),
        "lo": np.round(p0 * np.exp(f["band_q10"].values), 2).tolist(),
        "hi": np.round(p0 * np.exp(f["band_q90"].values), 2).tolist(),
    }
    eq = pd.DataFrame({"model": pnl, "tech": pnl_base, "bh": bh}).fillna(0)
    eq = (1 + eq).cumprod()
    edata = {"d": [str(d.date()) for d in eq.index], **{c: np.round(eq[c].values, 4).tolist() for c in eq}}
    qnames = {"cta_nvda_pos": "CTA 趋势基金:NVDA 估算仓位", "cta_ndx_pos": "CTA 趋势基金:纳指估算仓位",
              "volctl_expo": "波动率目标基金:股票敞口", "riskpar_lev": "风险平价基金:杠杆"}
    qdata = [{"name": qnames[c], "d": [str(d.date()) for d in quant.index],
              "y": np.round(quant[c].values, 3).tolist()} for c in quant]
    abl = [{"name": GROUP_ZH.get(a["group"], a["group"]), "ic_drop": a["ic_drop"]}
           for a in results.get("ablation", [])]
    payload = json.dumps({"h": h, "price": pdata, "equity": edata, "quant": qdata, "ablation": abl},
                         default=lambda x: None if x != x else x).replace("NaN", "null")

    rows = []
    for hh, d in results["horizons"].items():
        for label, key in (("全因子", "full_model"), ("仅技术面", "technical_only")):
            r = d[key]
            rows.append(
                f"<tr><td>{hh}日 · {label}</td><td>{r['n']}</td>"
                f"<td class='{_cls(r['oos_r2_vs_random_walk'])}'>{_pct(r['oos_r2_vs_random_walk'], 2)}</td>"
                f"<td>{_pct(r['price_mape_model'], 2)}</td><td>{_pct(r['price_mape_random_walk'], 2)}</td>"
                f"<td>{_pct(r['hit_rate'])}</td><td>{_num(r['hit_rate_pvalue'])}</td>"
                f"<td>{_pct(r['up_share_actual'])}</td><td class='{_cls(r['ic_spearman'])}'>{_num(r['ic_spearman'])}</td>"
                f"<td>{_num(r['monthly_ic_tstat'], 2)}</td>"
                f"<td>{_pct(r['coverage_80'])}</td><td>{_pct(r['coverage_95'])}</td></tr>")
    srows = "".join(
        f"<tr><td>{n}</td><td>{_pct(s['total_return'])}</td><td>{_pct(s['cagr'])}</td>"
        f"<td>{_pct(s['ann_vol'])}</td><td>{_num(s['sharpe'], 2)}</td><td>{_pct(s['max_drawdown'])}</td></tr>"
        for n, s in (("全因子 + 风控", st["risk_controlled_model"]), ("仅技术面 + 风控", st["technical_only"]),
                     ("买入持有", st["buy_and_hold"])))
    groups = "、".join(f"{GROUP_ZH.get(g, g)}({n})" for g, n in results["groups"].items() if g != "target")
    miss = results["missing_inputs"]

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>英伟达因子预测</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script></head><body><main>
<h1>英伟达(NVDA)外部因子预测 vs 现实</h1>
<p class="sub">样本外区间 {results['test_start']} 至 {results['last_date']} · {results['n_features']} 个量化因子 ·
每 21 个交易日滚动重训,训练集只用当时已知的数据</p>

<div class="tiles">
<div class="tile"><div class="k">{h}日方向命中率</div><div class="v">{_pct(m['hit_rate'])}</div>
<div class="d">实际上涨占比 {_pct(m['up_share_actual'])} · p={_num(m['hit_rate_pvalue'])}</div></div>
<div class="tile"><div class="k">信息系数 IC</div><div class="v">{_num(m['ic_spearman'])}</div>
<div class="d">月度 IC t 值 {_num(m['monthly_ic_tstat'], 2)}</div></div>
<div class="tile"><div class="k">样本外 R²(对比随机游走)</div><div class="v">{_pct(m['oos_r2_vs_random_walk'], 2)}</div>
<div class="d">>0 才算赢过"明天=今天"</div></div>
<div class="tile"><div class="k">80% 区间覆盖率</div><div class="v">{_pct(m['coverage_80'])}</div>
<div class="d">越接近 80% 风险刻画越准</div></div>
<div class="tile"><div class="k">风控策略夏普</div><div class="v">{_num(st['risk_controlled_model']['sharpe'], 2)}</div>
<div class="d">买入持有 {_num(st['buy_and_hold']['sharpe'], 2)}</div></div>
<div class="tile"><div class="k">最新预测({lf['date']} 起 {h} 日)</div><div class="v">${lf['price_mid']:.2f}</div>
<div class="d">80% 区间 ${lf['price_80'][0]:.2f} – ${lf['price_80'][1]:.2f}</div></div>
</div>

<h2>预测股价 vs 实际股价</h2>
<p class="note">每个点是 {h} 个交易日前用当时全部因子做出的预测,画在它预测的那一天上。阴影是风险模型给出的 80% 区间。</p>
<div class="card"><div id="c_price" class="chart"></div></div>

<h2>各预测期的准确度(对比基准)</h2>
<p class="note">"仅技术面"只用 NVDA 自身价格和财报日历,用来检验外部因子到底有没有增量价值。价格 MAPE 对比的是"未来价=今天价"的随机游走。</p>
<div class="card tw"><table><tr><th>模型</th><th>样本数</th><th>R² vs 随机游走</th><th>价格MAPE</th>
<th>随机游走MAPE</th><th>命中率</th><th>p 值</th><th>实际上涨占比</th><th>IC</th><th>月度IC t</th>
<th>80%覆盖</th><th>95%覆盖</th></tr>{''.join(rows)}</table></div>

<h2>按预测交易:风控策略 vs 买入持有</h2>
<p class="note">仓位 = 半凯利(预期收益 / 预测方差),VIX 期限结构倒挂时减半,上限 ±1 倍,每单位换手扣 5bp。</p>
<div class="card"><div id="c_eq" class="chart"></div></div>
<div class="card tw"><table><tr><th>策略</th><th>总收益</th><th>年化</th><th>年化波动</th><th>夏普</th><th>最大回撤</th></tr>{srows}</table></div>

{"" if not abl else """<h2>哪组外部因子真正有用</h2>
<p class="note">逐组剔除后重新做完整的滚动预测,看 IC 掉多少。负值说明这组因子在样本外反而是噪声。</p>
<div class="card"><div id="c_abl" class="chart"></div></div>"""}

<h2>量化机构行为估算(样本外区间)</h2>
<p class="note">用各类基金公开的交易规则,从价格反推它们的仓位:趋势 CTA(20/60/120/250 日动量,15% 波动目标)、
波动率目标基金(10% 目标)、风险平价(股债等风险,10% 目标)。</p>
<div class="grid3">{''.join(f'<div class="card"><div class="note">{q["name"]}</div><div id="c_q{i}" class="chart small"></div></div>' for i, q in enumerate(qdata))}</div>

<h2>方法与局限</h2>
<p class="note">因子组:{groups}。量化行为因子另含杠杆 ETF 收盘调仓压力、动量因子拥挤度、统计套利残差、期权到期/四巫日/月末再平衡日历、
隔夜 vs 日内收益拆分、Amihud 流动性。FRED 数据按发布延迟后移,防止"偷看未来"。
没有纳入的:真实期权链 Gamma 敞口、13F 持仓、做空余额、新闻情绪 — 这些需要付费或额外数据源。
{('本次缺失的输入:' + '、'.join(miss) + '。') if miss else ''}
股价预测本质是低信噪比问题,IC 0.02–0.05 在业内已属可用;本报告不构成投资建议。</p>
</main><script>{JS.replace('__DATA__', payload)}</script></body></html>"""
    path.write_text(html, encoding="utf-8")
