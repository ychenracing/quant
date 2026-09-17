# TechQuant：A 股科技收益优先的盘后决策辅助

TechQuant 是独立开发的现金多头、日频研究与人工决策支持系统。默认生产路径是 `passive_ownership`：对给定科技股票池分配初始资金，在满足下一交易日开盘、整手、容量和现金约束时完成初始持有，之后不主动卖出、不主动调仓。系统不连接券商、不自动下单。

当前生产选择以长期收益为最高优先级。风险、回撤和压力窗口仍完整披露，但不作为默认策略晋级的阻塞条件。账户现金守恒、禁止做空、无未来信息、盘后决策/下一交易日执行、费用/滑点、数据身份和证据完整性仍是硬约束。

## 已验证的收益

冻结研究区间为 2023-01-03 至 2026-09-11，初始资金 200 万元，34 只科技股，896 个交易日。默认 passive 路径在三个核心范围的扣成本终值财富为：

- 34 股并集：约 **9.1023 倍**；
- 共同五股：约 **29.0630 倍**；
- 同时移除三只光通信龙头：约 **5.7161 倍**。

生产适配器与同引擎 `buy_hold` 的固定 199-case 历史账户已经逐案绑定；净值、目标和完整订单账本在既有数值容差内等价。参考策略比较及风险局限见 [研究结果](docs/RESULTS.md)。这些历史结果不保证未来收益。

## 开始使用

```bash
python -m pip install -e .
python -m techquant --help
python -m unittest discover -s tests -v
```

恢复经过哈希核验的冻结输入后：

```bash
python -m techquant backtest --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --pool union --output runs/union
python -m techquant inspect --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --pool chatgpt_5 --end 2026-09-11
python -m techquant verify runs/union
```

默认读取 `config/production.json`。该文件只包含 passive 实际使用的资金与执行参数；趋势、止损和风险参数不会被生产 CLI 静默接受。研究基准可通过 `backtest --benchmark buy_hold` 或 `--benchmark equal_weight` 显式运行，其身份与生产策略分开记录。

`inspect` 用直白中文输出待买、继续持有、阻止成交原因、现金余额和风险暴露。它重放模型账本，不读取真实账户；执行前必须人工核对真实持仓、公司行动、停复牌、涨跌停、可卖数量和资金。

详细说明：[架构](docs/ARCHITECTURE.md)、[操作手册](docs/OPERATIONS.md)、[研究结果](docs/RESULTS.md)、[研究合同](docs/RESEARCH_CONTRACT.md)。
