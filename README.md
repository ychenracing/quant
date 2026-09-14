# TechQuant：A 股科技趋势研究与盘后决策辅助

独立开发的现金多头、日频研究系统。只使用 2023 年起的科技股数据；盘后观察，下一交易日才允许模拟执行。**当前经济验收未通过，不应替代已经验证的策略，也不提供自动下单或实盘账户托管。**

## 能做什么

使用同一套全局参数，完成趋势筛选、持仓迟滞、组合风险降档与分档恢复；对既定科技目录中的非空子集运行同一引擎。保存逐日净值、目标权重、成交和阻塞原因、参数、数据及源码身份。盘后检查逐标的解释模型持仓与目标，不把回放账本当作用户真实持仓。

实现只依赖 NumPy 和 pandas。四个参考项目不进入生产包；原生对照由独立进程只读运行。没有股票专用参数表、针对已知下跌日期的退出开关、杠杆、做空、券商接口或后台交易服务。

## 实测结论，而非宣传承诺

固定研究区间为 2023-01-03 至 2026-09-11，共 896 个交易日，覆盖四项目股票池的 34 股并集。12 个数值候选仅按 2023—2025 年选优；2026 年是已知历史压力测试，不是未触碰的样本外检验。

已复现的 210 场景、437 次回放中，197 个同执行买入持有比较只有 3 个同时实现收益不低、回撤不高。共同五股策略终值约 10.8248 倍，买入持有约 29.0630 倍；因此没有达到“收益至少持平最优、风险全面最早、任意组合全面超越”的目标。发布 main 仅表示交付研究代码，不表示经济能力验收通过。结果对应的具体源码、数据和限制见 [研究结果](docs/RESULTS.md)。

## 开始使用

在仓库根目录，使用 Python 3.11 或以上版本；严格复现实验使用 Python 3.13.5、NumPy 2.3.5、pandas 2.2.3。

```bash
python -m pip install -e .
python -m techquant --help
python -m unittest discover -s tests -v
```

先按 [操作手册](docs/OPERATIONS.md) 恢复经过哈希核验的 `inputs/market`、`inputs/supplement` 和 `inputs/indices`。冻结数据没有被重新下载到另一个口径下；不要把新行情覆盖旧研究输入。

```bash
python -m techquant backtest --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --config config/research.json --pool union --output runs/union
python -m techquant inspect --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --config config/research.json --pool chatgpt_5 --end 2026-09-11
python -m techquant verify runs/union
```

输出全部标记为研究用途。`verify` 校验文件完整性，不判定盈利能力。

## 阅读顺序

[架构、策略与全部参数](docs/ARCHITECTURE.md)；[操作、复现与故障处理](docs/OPERATIONS.md)；[实测、缺点与后续方向](docs/RESULTS.md)；[冻结研究合同](docs/RESEARCH_CONTRACT.md)。

常规 CI 只有一个离线正确性作业，超时上限 4 分钟，不要求经济矩阵或覆盖率门槛。完整研究是单独、显式触发的流程，超时上限 9 分钟；失败与超时证据也必须保留。完整研究输出追加到 `research/evidence` 分支，不通过改写源码分支来冒充同一 SHA 的验收结果。
