# Native inventory measurement bundle

This directory preserves the raw execution-correctness comparison for the workbuddy/common-five T+1 inventory reconciliation. It is not a strategy candidate and is not economic acceptance.

Reconstruct the archive by concatenating the eight UTF-8 base64 parts in lexical order and decoding the result:

```bash
cat native-inventory-measurement.tar.gz.b64.part-* | base64 -d > native-inventory-measurement.tar.gz
sha256sum native-inventory-measurement.tar.gz
```

Expected archive SHA256:

`dc75f8962f13c994c01e5038860a3ad03f7580abbf9af7371fd2365af7b76741`

Base64-part SHA256 values before decoding:

- `part-000`: `7e2b416db4c28920c3f7a14d4fe2e239f93b8811da2f6f36fd065999f519fb02`
- `part-001`: `c0f4087eb02def015f6ce2540b3ba325d228af53f421610e827ab6049d32881d`
- `part-002`: `7a9f7494094f508343e7c1b89163f09dc899556a3e2e3d912fb7c88ac11ad24f`
- `part-003`: `4e69f302e9510cd2589acf4512a61b1e53da262f0927600d6038c952ce1bb01f`
- `part-004`: `ade6454dd36a82ec2dfa7154ef2ba472ece246973e54779e062f5e78716d6821`
- `part-005`: `e524e9d086ff0c033650a72fe34848b5423a5f9169848079f4f02e93000633f8`
- `part-006`: `204acb9ac93f7af714ba0222e2a112c6d5187f6715e5cc7945f4586084ac0a6d`
- `part-007`: `71023e149e455b10a371537ed2f2c510fa6794210b5e55a6de3c8450c6576993`

The archive contains the control and corrected curves, fills, settings, source identities/manifests, logs, audit output and verifier. It deliberately excludes the private reference source and the private corrected source/patch. Their hashes are recorded in `provenance.json` instead.

Key raw-content hashes include the exact control trades SHA256 `a9620cef60d7f18623af3d676ac7aaaa29d7de9613712812764998eecb406adb` and corrected trades SHA256 `7befacaa4c82792aa6d380c1412fdce812830e1d126ee15ba895e1721a668db0`.
