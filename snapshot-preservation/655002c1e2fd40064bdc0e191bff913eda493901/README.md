# 快照修正与本地原始验证保全

本目录保留先前因直接 Git DNS 失败未推送的原始本地提交、补丁及验证记录。
插件发布源码为 `655002c1e2fd40064bdc0e191bff913eda493901`，与原本地 e8dddd7 的树完全一致。
清单和接收记录不将工程通过改写成经济验收，也不声称其他容器原件全部可见。

在空目录恢复：

```bash
sha256sum -c SHA256SUMS
cat snapshot-preservation.tar.xz.part-* > snapshot-preservation.tar.xz
# 核对 receipt.json 的 archive_sha256 后再解压。
tar -xJf snapshot-preservation.tar.xz
```

解压后核验 `PRESERVATION_MANIFEST.json`，原 Git bundle 按其说明恢复。不要覆盖现有工作区。
本次固定候选完整覆盖原始账本另见 `fixed-validation/e7341fc52e4a11cc4d8396aa3f15819f50702877/34876226498/`。
