# 操作、复现与故障处理

## 环境与冻结输入

在仓库根目录执行命令。一般运行要求 Python >=3.11；冻结研究环境为 Python 3.13.5、NumPy 2.3.5、pandas 2.2.3。外部原生对照另需要 PyYAML 6.0.3；它不是生产依赖。

```bash
python -m pip install -e .
python -m pip install PyYAML==6.0.3
```

`research/run-request.json` 记录三个冻结压缩包的 artifact ID、SHA256 和参考源码 SHA。首次输入是附件中的 `quant-market-evidence.zip`、`quant-market-supplement.zip`、`quant-frozen-reference-indices.zip`。分别对应 `market.zip`、`supplement.zip`、`indices.zip`；改文件名不改内容。

完整成功发布后的耐久恢复入口为 `research/evidence` 分支的 `LATEST.json`。首先核验其中的 `source_commit` 是否正是需要检查的源码；不能用 LATEST 指向的旧提交结果证明当前 main。其 `published/<source_sha>/<run_id>/` 目录保存分卷证据、`SHA256SUMS`、`receipt.json`、报告、源码身份和总清单。

```bash
sha256sum -c SHA256SUMS
cat evidence.tar.gz.part-* > evidence.tar.gz
# 对照 receipt.json 中 archive_sha256，再解压到新的空目录。
tar -xzf evidence.tar.gz
```

完整包中的 `evidence/inputs/` 可直接作为回测输入；`frozen_archives/` 可重新用于完整复现。仅解压来自可信来源、哈希匹配的包，不把不可信压缩包覆盖源码目录。GitHub Actions 临时附件会到期，耐久证据分支不依赖这些附件的有效期；重新执行 Actions 时若旧 artifact 已到期，应从已核验的证据包恢复原压缩包，而不是重新下载市场并冒充冻结输入。

## 常用命令

```bash
python -m techquant audit --data inputs/market --supplement inputs/supplement --catalog research/catalog.json
python -m techquant backtest --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --config config/research.json --pool union --output runs/union
python -m techquant backtest --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --config config/research.json --symbols sz300308,sh688008,sh603986 --output runs/custom
python -m techquant inspect --data inputs/market --supplement inputs/supplement --catalog research/catalog.json --config config/research.json --pool chatgpt_5 --end 2026-09-11
python -m techquant verify runs/union
```

`--pool` 与 `--symbols` 二选一，标的须在已审查科技目录和数据中存在。重复标的、空池、无覆盖、重复日期和文件损坏报错退出，不改股票池凑出结果。股票池支持任意非空子集的输入，不代表任意子集都会盈利或最优。

`inspect` 输出截至所指定收盘日的模型持仓、目标、趋势条件、风险解释和各标的说明。它重新回放模型，不读取或写入真实账户。不产生订单文件；人工必须重新核对实际持仓、公司行动、可卖股数、停牌、涨跌停、资金、权限和下一交易日。冻结数据截至 2026-09-11；更晚的研究需要独立的新数据快照，不能把旧报告称作实时信号。

`backtest` 输出目录必须不存在。`equity.csv` 为逐日净值和风险原因；`targets.csv` 为目标权重；`orders.csv` 为成交与阻塞尝试；`identity.json` 绑定配置、行情、股票池、源码、runtime、成本与延迟；`metrics.json` 为指标；`manifest.json` 保存文件哈希。空成交文件也必须存在。

## 固定选优与完整评估

```bash
# 在干净的真实源码 checkout 中声明实际提交，不得填写另一个 SHA。
export QUANT_SOURCE_COMMIT="$(git rev-parse HEAD)"
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python research/study.py select --data inputs/market --supplement inputs/supplement --output study/select
python research/study.py evaluate --data inputs/market --supplement inputs/supplement --selection study/select/selection.json --output study/evaluate --batch-size 250
```

中断的评估可使用原命令加 `--resume`。只有源码、runtime、数据、runner、冻结协议和选优配置等身份一致才允许复用；不一致时新建输出目录重跑，不能编辑旧 identity 让校验通过。选择不支持覆盖旧输出或悄悄续写另一组候选。

完整原生对照必须是只读参考仓库，固定到 `research/run-request.json` 指定的 reference_commit。它不是生产依赖。参考 checkout、三个压缩包准备后：

```bash
python research/reproduce.py --archives archives --reference ../trades --output result
```

需要复现历史父模型时，另给 `--parent`，其 SHA 必须匹配 request。完整 runner 要求真实、干净且声明一致的 Git checkout；它检查数据指纹，运行冻结候选与全部有限场景，原生对照逐项保留成功、覆盖失效、异常和超时，最后生成总清单及分离收据。

`result/execution.json` 的执行完成不等于经济通过。比较风险时同时看信号日、关联成交日与实际敞口；不能拿第一次降预算日期就宣称早于所有对照发现风险。

## CI 与故障处理

日常正确性只跑离线测试、编译和 CLI 帮助，作业最多 4 分钟。完整研究流程是 `Explicit research reproduction`，不是常规经济门禁；只允许手动触发或 main 上显式研究请求文件变化触发，最多 9 分钟，不定时轮询。

输入 SHA 不符：隔离该文件，从耐久证据恢复准确原件。参考覆盖失败：保留原生实际日期，不补零或前向填充净值冒充全区间。原生预算耗尽：保留超时和日志，不认定策略亏损或胜出；本地单独复现该外部实现，不修改其策略。经济失败：查看 matrix、原生报告和风险时点，不扩大既定搜索来拟合已知 2026 路径。必要正确性失败：先定位并修复，不能跳过校验或改掉冻结标准。
