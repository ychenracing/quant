# 公开研究流程的权限与证据范围

首次 main 交付 `ce25f1d67aae8ccac18c3b89c65169d1f85a4788` 的离线 CI 已通过，完整研究却在读取私有 trades 仓库时失败。quant 的默认 GITHUB_TOKEN 仅限本仓库，不能读取另一私有仓库。没有扩大凭据权限，也没有把私有参考源码复制到公开项目。

公开流程现在明确使用 `--without-native`，继续从当前实际源码重新生成全部 12 候选、210 场景、437 次回放，以及失败父版本对照。27 项原生计划逐项记录为 `REFERENCE_CHECKOUT_UNAVAILABLE`，没有任何虚构的收益或回撤；不会被计为回放成功或竞争胜出。完整原生回放仍使用操作手册中的授权本地 `--reference` 入口。

本地已经完成的原生测量按原生产者 SHA 保留在 docs/RESULTS.md、evidence/reproduced_567a2b6.json 及历史完整证据中。它们不是新 main 的重跑证明。公开产物中的 `SOURCE_COMPLETED_NATIVE_UNAVAILABLE` 表示本次源码研究完成而原生部分未运行，不能缩写成全面经济验收完成。

本修复只改变研究编排和权限失败披露，不改变生产策略、市场快照、费用、收益定义、冻结候选或经济验收。新增一项小型回归确保所有 27 个计划都被保留且没有伪造指标；必要离线测试现在共 39 项。

依据：https://docs.github.com/en/actions/concepts/security/github_token 。
