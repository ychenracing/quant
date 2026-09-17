# 操作、复现与故障处理

## 环境与冻结输入

一般运行要求 Python >=3.11；冻结研究环境为 Python 3.13.5、NumPy 2.3.5、pandas 2.2.3。安装：

```bash
python -m pip install -e .
```

冻结输入通过 `research/run-request.json` 和耐久证据绑定。复现历史经济结果时必须核对源码 SHA、数据哈希和运行身份，不能重新下载更新后的行情替换冻结输入。已验证冻结数据指纹为 `d9ded1d933f292933aa09145bc36a5696421c43dbf266fce018a34ef0b9f3f9b`。

## 默认生产命令

```bash
python -m techquant audit --data inputs/market --supplement inputs/supplement --catalog research/catalog.json
python -m techquant backtest --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --pool union --output runs/union
python -m techquant inspect --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --pool chatgpt_5 --end 2026-09-11
python -m techquant verify runs/union
```

默认策略为 `passive_ownership`，默认配置为 `config/production.json`。`--pool` 和 `--symbols` 二选一，标的必须属于已审查科技目录。重复标的、数据损坏、越界日期或未支持的生产配置字段均失败退出。

`inspect` 输出：待继续买入的初始预算、继续持有、阻止成交原因、现金余额、模型净值、敞口和历史最大回撤。默认策略没有主动卖出建议；这不是遗漏，而是策略语义。输出只供人工复核，不生成或发送券商订单。

研究 benchmark 必须显式声明：

```bash
python -m techquant backtest ... --benchmark buy_hold --output runs/buy_hold
python -m techquant backtest ... --benchmark equal_weight --output runs/equal_weight
```

benchmark 身份保存在结果元数据中，不能改名为生产策略结果。

## 输出与核验

`backtest` 输出目录不可覆盖既有结果。`equity.csv` 保存逐日账户，`targets.csv` 保存目标，`orders.csv` 保存成交和阻塞尝试，`identity.json` 绑定源码、数据、股票池、成本和执行假设，`metrics.json` 保存指标，`manifest.json` 保存文件哈希。`verify` 只验证完整性，不判定未来盈利。

生产 passive 已与固定 199-case 历史 `buy_hold` 账户逐案核验。比较保留日期、方向、状态、阻止原因、数量、费用、现金、净值和目标；只对 CSV 浮点往返误差使用既有数值容差。

## 故障处理

输入 SHA 不符时隔离输入并恢复准确原件。开盘不可用、涨跌停近似、容量不足、整手/现金不足或裁剪后金额过小时，订单记录为 BLOCKED，不能制造成交。研究参考覆盖不足或执行超时保持为未验证，不能当作本策略胜出。

常规 CI 只运行短离线正确性、编译和 CLI 帮助。完整历史经济研究不放进日常 CI。
