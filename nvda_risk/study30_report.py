"""HTML page for the 30-minute study."""
from report import CSS, _num, _pct
from zoo import FAMILY, LABEL, MODELS

EXTRA = """main{max-width:1200px}table{font-size:12.5px}th,td{padding:6px 7px}
th.grp{text-align:center;border-bottom:1px solid var(--axis);color:var(--ink)}
tr.pick td{background:color-mix(in srgb,var(--s1) 10%,transparent)}td.fam{color:var(--muted)}
.verdict{font-size:17px;line-height:1.7}"""


def _c(x):
    return "good" if x > 0 else "bad"


def _verdict(res):
    a, b, c = res["A"], res["B"], res["C"]
    lines = []
    for k, s in res.items():
        p = s["models"][s["cv_pick"]]
        rc = s["reality_check"]
        sig = p["hit_p"] < 0.05 or abs(p["ic_t"]) >= 2
        lines.append(
            f"<b>{s['title']}</b>:训练期交叉验证选出「{LABEL[s['cv_pick']]}」,样本外 IC {p['ic']:+.3f}(t={p['ic_t']:.2f}),"
            f"命中率比基准 {p['hit_edge'] * 100:+.1f} 个百分点(p={p['hit_p']:.3f})"
            + ("" if sig else ",不显著")
            + f"。全部 {len(s['models'])} 个模型里最好的那个能否用运气解释:命中率 p={rc['rc_p_hit']:.2f}、IC p={rc['rc_p_ic']:.2f}"
            + ("(都没过 0.05,最好的结果也可能是运气)。" if min(rc['rc_p_hit'], rc['rc_p_ic']) >= 0.05 else "。"))
    pos = {k: sum(m["ic"] > 0 for m in s["models"].values()) for k, s in res.items()}
    lines.append("<b>多少模型的样本外 IC 为正:</b>" + "、".join(f"{k} 组 {pos[k]}/{len(res[k]['models'])}" for k in res)
                 + "。如果真有稳定信号,大部分合理的模型都应该能抓到一些。")
    ps = c.get("per_stock", {})
    if ps:
        pick = c["cv_pick"]
        good = [t for t, d in ps.items() if d[pick]["ic"] > 0]
        lines.append(f"<b>能否推广到其他股票:</b>合并模型在 9 只股票上 IC 为正的有 {len(good)} 只({'、'.join(good) or '无'});"
                     f"NVDA 自己 {ps['NVDA'][pick]['ic']:+.3f}。")
    def passes(s):
        p, rc = s["models"][s["cv_pick"]], s["reality_check"]
        return ((p["hit_p"] < 0.05 and rc["rc_p_hit"] < 0.05 and rc["best_hit_model"] == s["cv_pick"])
                or (rc["single_p_ic"][s["cv_pick"]] < 0.05 and rc["rc_p_ic"] < 0.05))
    promising = sum(passes(s) for s in res.values()) >= 2      # must replicate in a second setup
    # leads: positive in the training-period CV *and* out of sample, in every setup
    leads = [k for k in MODELS if all((s["cv_scores"][k] or 0) > 0 and s["models"][k]["ic"] > 0 for s in res.values())]
    if leads:
        lines.append("<b>在三组里训练期和样本外 IC 都为正的模型:</b>"
                     + "、".join(f"{LABEL[k]}(样本外 IC " + " / ".join(f"{res[g]['models'][k]['ic']:+.3f}" for g in res)
                                 + ",单独检验 p " + " / ".join(f"{res[g]['reality_check']['single_p_ic'][k]:.2f}" for g in res) + ")"
                                 for k in leads)
                     + "。这是比单组最好成绩更可靠的线索,但单独检验的 p 值需要都 <0.05 才算站得住。")
    head = ("<p class='verdict'><b>结论:有希望,值得继续跟踪。</b>至少两组在事先选定模型和全模型运气检验下都站得住。</p>"
            if promising else
            "<p class='verdict'><b>结论:目前不能算有希望。</b>扩大数据、换了 17 种模型之后,上一轮看到的 30 分钟信号没有在第二组数据里重现,"
            "最好的结果也通不过运气检验。</p>")
    return head + "<ul class='concl'>" + "".join(f"<li>{x}</li>" for x in lines) + "</ul>"


def _model_table(s):
    rows = []
    for name in MODELS:
        m = s["models"][name]
        cls = " class='pick'" if name == s["cv_pick"] else ""
        rows.append(f"<tr{cls}><td>{LABEL[name]}{' ★' if name == s['cv_pick'] else ''}</td><td class='fam'>{FAMILY[name]}</td>"
                    f"<td>{_num(s['cv_scores'][name])}</td>"
                    f"<td class='{_c(m['ic'])}'>{_num(m['ic'])}</td><td>{_num(m['ic_t'], 2)}</td>"
                    f"<td>{_pct(m['hit'])}</td><td class='{_c(m['hit_edge'])}'>{m['hit_edge'] * 100:+.1f}</td>"
                    f"<td>{_num(m['hit_p'], 3)}</td><td>{_num(s['reality_check']['single_p_ic'][name], 3)}</td>"
                    f"<td>{m['edge_bp']:+.2f}</td>"
                    + (f"<td>{_num(m['xs_ic'])}</td><td>{_num(m['xs_ic_t'], 2)}</td>" if "xs_ic" in m else "")
                    + "</tr>")
    xs = "xs_ic" in next(iter(s["models"].values()))
    head = ("<tr><th>模型</th><th>类型</th><th>训练期 CV IC</th><th>样本外 IC</th><th>IC t</th><th>命中率</th>"
            "<th>超出基准(百分点)</th><th>命中 p</th><th>IC p(按天聚类)</th><th>每笔超额 bp</th>"
            + ("<th>截面 IC</th><th>截面 t</th>" if xs else "") + "</tr>")
    rc = s["reality_check"]
    first = next(iter(s["models"].values()))
    return (f"<h2>{s['title']}</h2><p class='note'>{s['rows']:,} 行 · {s['n_features']} 个因子 · {s['days']} 个交易日 · "
            f"样本外 {s['test_window']}(样本 {first['n']:,},有效 {first['n_eff']:,})· 基准命中率(总猜多数方向){_pct(first['base'])} · "
            f"★ = 只用训练期交叉验证选出的模型 · 运气检验:最好命中率「{LABEL[rc['best_hit_model']]}」p={rc['rc_p_hit']:.2f},"
            f"最好 IC「{LABEL[rc['best_ic_model']]}」p={rc['rc_p_ic']:.2f}</p>"
            f"<div class='card tw'><table>{head}{''.join(rows)}</table></div>")


def _stock_table(c):
    pick = c["cv_pick"]
    rows = "".join(
        f"<tr><td>{t}</td><td class='{_c(d[pick]['ic'])}'>{_num(d[pick]['ic'])}</td><td>{_num(d[pick]['ic_t'], 2)}</td>"
        f"<td class='{_c(d[pick]['hit_edge'])}'>{d[pick]['hit_edge'] * 100:+.1f}</td>"
        f"<td class='{_c(d['lgbm_small']['ic'])}'>{_num(d['lgbm_small']['ic'])}</td>"
        f"<td class='{_c(d['lgbm_small']['hit_edge'])}'>{d['lgbm_small']['hit_edge'] * 100:+.1f}</td>"
        f"<td class='{_c(d['rule_reversal']['ic'])}'>{_num(d['rule_reversal']['ic'])}</td></tr>"
        for t, d in c["per_stock"].items())
    return (f"<h2>合并模型在每只股票上的表现</h2><div class='card tw'><table>"
            f"<tr><th></th><th class='grp' colspan='3'>CV 选出的「{LABEL[pick]}」</th><th class='grp' colspan='2'>LightGBM 小(上一轮 NVDA 选的)</th><th class='grp'>规则:短期反转</th></tr>"
            "<tr><th>股票</th><th>IC</th><th>IC t</th><th>命中率超出</th><th>IC</th><th>命中率超出</th><th>IC</th></tr>"
            f"{rows}</table></div>")


def build(res, cost, path):
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>30 分钟信号检验</title>
<style>{CSS}{EXTRA}</style></head><body><main>
<h1>30 分钟信号是真的吗?</h1>
<p class="sub">上一轮 NVDA 30 分钟预测的命中率比基准高 5.8 个百分点(p=0.015),但样本只有一个月。这里用三种方式扩大数据、
换 {len(MODELS)} 种模型重新检验。所有组都用同一个样本外区间(数据的后一半),每组分 10 段滚动重训。</p>
<div class="card">{_verdict(res)}</div>
{''.join(_model_table(s) for s in res.values())}
{_stock_table(res['C']) if res['C'].get('per_stock') else ''}
<h2>怎么读</h2>
<p class="note">训练期 CV IC = 只用训练期数据做交叉验证的成绩,用来事先选模型(★),这是唯一不受"挑结果"影响的数字。
其他模型的样本外成绩只是参考:模型越多,总会有几个碰巧好看。运气检验(White Reality Check,按交易日分块自助法 1000 次)
回答的就是"在这么多模型里,最好的那个有多大可能纯靠运气"。命中 p 和 IC t 用有效样本(5 分钟组的 30 分钟收益互相重叠,按 ÷6 计);IC p(按天聚类)是单个模型按交易日分块自助法的 p 值,
它考虑了同一天里各股票、各时段一起波动,比 IC t 更保守。预测为 0 的时点(规则模型只在特定时段出手)不计入命中率。
每笔超额 = 按预测方向交易一次的平均收益减一直持有,未扣成本(来回约 {cost:g}bp)。截面 IC = 同一时刻 9 只股票之间的排序能力。
不构成投资建议。</p>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
