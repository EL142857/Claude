# Alpaca 模拟盘

## 多股票 × 多规则(`multi.py`,当前在跑的版本)

45 只大盘股(`rules.UNIVERSE`)× 8 条不用训练的规则(`rules.RULES`):30 分钟反转、残差反转(剔除 QQQ)、
截面反转、大幅波动才反转、60 分钟反转、开盘动量→最后半小时、跳空回补、随机(安慰剂)。
每条规则每只股票约 $1,000(截面反转 $2,000);所有规则按股票净额后只下净差额的单,总敞口上限 $35 万;
每条规则的盈亏用虚拟账本单独计算,成本按真实成交价的滑点分摊(`multi_eval.py`)。

```bash
python paper/backtest.py               # 全部规则在 Yahoo 30 分钟 K 线上回测(和实盘同一套决策代码)
python paper/multi.py --simulate 11:00 # 用今天的真实数据模拟一次决策,不下单
python paper/multi.py --dry-run        # 全天只记录不下单
python paper/multi.py                  # 跑当天的模拟盘
python paper/multi_eval.py             # 每条规则每天的盈亏、成本、命中率,以及和账户权益的核对
python paper/multi.py --flatten        # 紧急全部平仓
```

## 单只 NVDA(`bot.py`,第一版)

规则(与 `study30.py` 里的「规则:短期反转」完全一致,`--replay` 已核对 330/330 个决策方向相同):
每个交易日 10:30–15:30(美东)每半小时看过去 30 分钟涨跌,涨了就做空、跌了就做多,持有到下一个半点;
15:59 全部平仓,不过夜。每次仓位约 `PAPER_NOTIONAL` 美元(默认 1 万)。

## 准备

在云环境设置里添加环境变量(用 Alpaca **模拟盘 / paper** 账户的 key):

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`
- 可选:`PAPER_NOTIONAL`(每次仓位美元,默认 10000)、`ALPACA_FEED`(默认 `iex`,免费数据)

交易地址写死为 `https://paper-api.alpaca.markets`,不会碰实盘。

## 运行

```bash
python paper/bot.py --replay    # 用缓存的历史 K 线核对决策逻辑(不需要 key)
python paper/bot.py --dry-run   # 连上账户,只记录决策,不下单
python paper/bot.py             # 跑当天的模拟盘(开盘前启动,收盘后自动结束)
python paper/evaluate.py        # 汇总:真实成交 vs 回测假设价格、滑点、每日权益
touch paper/STOP                # 随时停止
```

在自己电脑上每天自动跑(cron,时区设为美东):

```
25 9 * * 1-5  cd /path/to/repo/nvda_risk && python paper/bot.py >> paper/logs/bot.out 2>&1
```

日志在 `paper/logs/YYYY-MM-DD.csv`:每个决策的前后价格、目标仓位、订单、成交价、账户权益。
