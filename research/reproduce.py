"""Explicit, finite research reproduction; never an ordinary CI economic gate.

Only the predeclared numerical grid is selected. Native references are read-only
processes, not implementation dependencies. Failed and partial runs are retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd

from techquant.data import file_hash, load_market
from techquant.evidence import source_identity, verify_evidence
from research.native import DIRECTORIES, reference_files


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def window_metrics(nav: pd.Series, start: str | None, end: str | None,
                   initial: float = 2_000_000.) -> dict:
    selected = nav.loc[start:end]
    if selected.empty:
        return {'status': 'NO_COVERAGE'}
    first = nav.index.get_loc(selected.index[0])
    base = float(nav.iloc[first - 1]) if first else initial
    path = np.r_[base, selected.to_numpy(dtype=float)]
    return {'status': 'MEASURED', 'wealth': float(path[-1] / base),
            'max_drawdown': float((1 - path / np.maximum.accumulate(path)).max()),
            'sessions': len(selected)}


def restore_archives(archives: Path, request: dict, destination: Path) -> None:
    for name, record in request['archives'].items():
        archive = archives / (name + '.zip')
        if file_hash(archive) != record['sha256']:
            raise ValueError(f'frozen archive identity mismatch: {name}')
        target = destination / name
        target.mkdir(parents=True, exist_ok=False)
        with ZipFile(archive) as z:
            for entry in z.infolist():
                resolved = (target / entry.filename).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError('unsafe archive member')
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('archive symlinks are not allowed')
            z.extractall(target)


def command(args: list[str], *, cwd: Path, log: Path, timeout: int,
            commit: str) -> int:
    env = dict(os.environ, QUANT_SOURCE_COMMIT=commit,
               PYTHONPATH=str(cwd / 'src') + os.pathsep + str(cwd),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with log.open('w', encoding='utf-8') as stream:
        try:
            completed = subprocess.run(args, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return completed.returncode
        except subprocess.TimeoutExpired:
            stream.write('\nRESEARCH_EXECUTION_BUDGET_EXCEEDED\n')
            return 124


def study(root: Path, data: Path, output: Path, commit: str) -> None:
    common = ['--data', str(data / 'market'), '--supplement', str(data / 'supplement')]
    for mode, extra in [('select', []), ('evaluate', ['--selection', str(output / 'select/selection.json'),
                                                   '--batch-size', '250'])]:
        result = command([sys.executable, 'research/study.py', mode, *common, *extra,
                          '--output', str(output / mode)], cwd=root, log=output / (mode + '.log'),
                         timeout=210, commit=commit)
        if result != 0:
            raise RuntimeError(f'{mode} execution failed ({result}); log and partial runs retained')


def native_cases(root: Path, data: Path, reference: Path, output: Path,
                 commit: str, catalog: dict) -> None:
    custom = json.loads(json.dumps(catalog))
    custom['pools']['remove_optical_leaders'] = [s for s in catalog['pools']['union']
                                               if s not in {'sz300308', 'sz300502', 'sz300394'}]
    catalog_path = output / 'catalog.json'
    write_json(catalog_path, custom)
    plan = []
    for ref in DIRECTORIES:
        scopes = ['chatgpt_5', 'union', 'remove_optical_leaders']
        scopes += [p for p in catalog['pools'] if p.startswith(ref + '_')]
        plan += [(ref, pool) for pool in dict.fromkeys(scopes)]
    write_json(output / 'case_plan.json', plan)
    records = {ref: reference_files(reference / directory) for ref, directory in DIRECTORIES.items()}
    write_json(output / 'reference_source_files.json', records)

    def one(case: tuple[str, str]) -> dict:
        ref, pool = case
        name = ref + '__' + pool
        target = output / name
        status = command([sys.executable, 'research/native.py', '--reference', ref,
                          '--reference-root', str(reference), '--data', str(data / 'market'),
                          '--supplement', str(data / 'supplement'), '--indices', str(data / 'indices'),
                          '--catalog', str(catalog_path), '--pool', pool, '--output', str(target)],
                         cwd=root, log=output / (name + '.log'),
                         timeout=180 if pool == 'chatgpt_5' else 60, commit=commit)
        measured = target / 'summary.json'
        value = json.loads(measured.read_text()) if measured.exists() else {
            'status': 'EXECUTION_BUDGET_EXCEEDED' if status == 124 else 'EXECUTION_FAILED'}
        row = {'reference': ref, 'pool': pool, 'process_return_code': status,
               'execution_budget_seconds': 180 if pool == 'chatgpt_5' else 60, **value}
        write_json(output / (name + '.outcome.json'), row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    # Independent processes isolate imported reference modules. Two workers bound
    # CPU/memory use; their order cannot affect selection or trading decisions.
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(one, plan))
    if records != {ref: reference_files(reference / directory) for ref, directory in DIRECTORIES.items()}:
        raise RuntimeError('reference executable/configuration files changed')
    write_json(output / 'outcomes.json', outcomes)


def report(root: Path, output: Path) -> dict:
    protocol = json.loads((root / 'research/protocol.json').read_text())
    evaluation = output / 'current/evaluate'
    matrix = pd.read_csv(evaluation / 'matrix.csv')
    summary = json.loads((evaluation / 'summary.json').read_text())
    selection = json.loads((output / 'current/select/selection.json').read_text())
    outcomes_path = output / 'native/outcomes.json'
    outcomes = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else []
    comparisons = []
    for row in outcomes:
        value = {'reference': row['reference'], 'pool': row['pool'], 'status': row['status']}
        if row['status'] == 'NATIVE_REPLAY_MEASURED':
            folder = output / 'native' / (row['reference'] + '__' + row['pool'])
            nav = pd.read_csv(folder / 'equity.csv', index_col=0, parse_dates=True)['nav']
            value['windows'] = {name: window_metrics(nav, *window)
                                for name, window in protocol['time_windows'].items()}
            value['native_trade_rows'] = row['orders_native_rows']
        comparisons.append(value)
    primary = matrix[(matrix['case'].isin(['union', 'chatgpt_5', 'remove_optical_leaders'])) &
                     (matrix['window'] == 'full')].to_dict('records')
    risk = {}
    for case in ('union', 'chatgpt_5', 'remove_optical_leaders'):
        folder = evaluation / 'runs' / (case + '_strategy')
        frame = pd.read_csv(folder / 'equity.csv', index_col=0, parse_dates=True)
        orders = pd.read_csv(folder / 'orders.csv')
        drops = (frame.target_cap.diff() < -1e-10) & (frame.index >= '2026-06-22') & (frame.index <= '2026-08-31')
        dates = frame.index[drops]
        signals = [{'signal_date': str(date.date()), 'reason': str(frame.loc[date, 'reason']),
                    'target_cap': float(frame.loc[date, 'target_cap'])} for date in dates]
        first = signals[0]['signal_date'] if signals else None
        sells = orders[(orders.status == 'FILLED') & (orders.side == 'SELL') &
                       (orders.signal_date.isin([s['signal_date'] for s in signals]))]
        risk[case] = {'budget_reductions': signals, 'first_budget_reduction': first,
                      'first_linked_sell': str(sells.iloc[0]['date']) if len(sells) else None,
                      'interpretation': 'budget alarms, not all risk signals; next-open fills are not prior knowledge'}
    result = {'economic_acceptance': 'FAILED_TO_ESTABLISH_REQUESTED_DOMINANCE',
              'engineering_and_publication_do_not_override_economics': True,
              'selection': selection, 'finite_evaluation': summary, 'primary_results': primary,
              'native_comparisons': comparisons, 'risk_timing': risk,
              'limitations': ['adjusted-unit proxy, not a corporate-action cash/share/tax ledger',
                              '2026 was already known; not an untouched prospective holdout',
                              'finite nonuniform cases do not establish all 2^34-1 subsets',
                              'native execution/fees/holdings calendars differ; not normalized superiority',
                              'no audited common risk-event labels, so earliest-risk dominance is unverified']}
    write_json(output / 'report.json', result)
    lines = ['# 量化研究实测报告', '', '**结论：未证明达到用户要求的全面超越，不能标记为经济验收通过。**', '',
             f"源码：`{selection['source']['commit']}`；包哈希：`{selection['source']['package_sha256']}`。", '',
             '范围：2023-01-03 至 2026-09-11；34 只科技股；盘后信号、下一交易日执行。', '',
             f"固定数值候选 12 个，选择候选 {selection['candidate']}；完整评估 {summary['case_count']} 场景、{summary['replay_count']} 次回放。", '',
             f"主要买入持有比较 {summary['matched_buy_hold_cases']} 例；收益不低于对照 {summary['wealth_at_least_buy_hold']} 例，收益与回撤同时占优 {summary['both_return_and_drawdown_dominate']} 例。", '',
             '## 核心同执行比较', '', '|组合|策略|终值财富倍数（含本金）|最大回撤|成交笔数|', '|---|---|---:|---:|---:|']
    for row in primary:
        lines.append(f"|{row['case']}|{row['policy']}|{row['wealth']:.6f}|{row['max_drawdown']:.2%}|{row['orders']}|")
    lines += ['', '## 四项目原生共同五股对照', '', '只统一数据、股票池、区间与初始资金；不把不同成交规则、费用或原生日内保护写成同执行胜出。', '',
              '|原生项目|状态|全期财富|全期回撤|', '|---|---|---:|---:|']
    for row in comparisons:
        if row['pool'] != 'chatgpt_5':
            continue
        measured = row.get('windows', {}).get('full', {})
        lines.append(f"|{row['reference']}|{row['status']}|{measured.get('wealth', '未验证')}|{measured.get('max_drawdown', '未验证')}|")
    lines += ['', '## 风险与执行', '', '详见 report.json 的逐次降险信号日与实际关联卖出日。模型不能在收盘前使用当日收盘信号，也不能保证跳空或跌停日卖出。', '',
              '## 复现与限制', '', 'current/ 保存全部候选、股票池/逐一移除/联合移除/行业移除/随机组合/全部数量/成本/延迟/参数邻域结果；parent/ 保存失败父版本对照。',
              'native/ 同时保存成功、覆盖失效、错误或执行预算耗尽；不删除失败条目。原生交易记录行数不冒充统一成交口径。',
              '数据与原始压缩包保存在 inputs/、frozen_archives/；SOURCE_IDENTITY.json、MANIFEST.json 绑定源码、数据、runner 与产出。',
              '34 只标的的全部非空组合为 17,179,869,183 个，本研究没有穷举。2026 年为已知历史压力检验，不能描述成未触碰样本。',
              '回测以复权经济单位记账，不是实股公司行动账本；股票池存在事后选择风险。风险识别全面最早、所有组合收益最大回撤最小均未验证。',
              '这些结果不构成实盘交易建议，也不证明未来收益。', '']
    (output / 'RUN_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = json.loads((root / 'research/run-request.json').read_text())
    commit = os.environ.get('QUANT_SOURCE_COMMIT', '')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if commit != actual or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise ValueError('reproduction requires a clean checkout of the declared source commit')
    actual_reference = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.reference, text=True).strip()
    if actual_reference != request['reference_commit']:
        raise ValueError('reference checkout differs from the frozen commit')
    write_json(output / 'SOURCE_IDENTITY.json', {'source': source_identity(), 'request': request,
               'reference_commit': actual_reference, 'reproducer_sha256': file_hash(Path(__file__)),
               'native_harness_sha256': file_hash(root / 'research/native.py'),
               'study_runner_sha256': file_hash(root / 'research/study.py'),
               'github_run_id': os.environ.get('GITHUB_RUN_ID'), 'github_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT')})
    data = output / 'inputs'
    restore_archives(args.archives.resolve(), request, data)
    catalog = json.loads((root / 'research/catalog.json').read_text())
    market = load_market(data / 'market', supplement=data / 'supplement', sectors=catalog['sectors'])
    if market.fingerprint() != request['data_sha256']:
        raise ValueError('restored data fingerprint differs from frozen market')
    shutil.copytree(args.archives, output / 'frozen_archives')
    subprocess.run(['git', 'archive', '--format=zip', '-o', str(output / 'source.zip'), 'HEAD'], cwd=root, check=True)
    for name in ('current', 'native'):
        (output / name).mkdir()
    study(root, data, output / 'current', commit)
    if args.parent:
        parent = args.parent.resolve()
        parent_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent, text=True).strip()
        if parent_sha != request['parent_commit']:
            raise ValueError('wrong preserved parent source')
        (output / 'parent').mkdir()
        study(parent, data, output / 'parent', parent_sha)
    native_cases(root, data, args.reference.resolve(), output / 'native', commit, catalog)
    for path in output.glob('*/**/runs/*/manifest.json'):
        verify_evidence(path.parent)
    report(root, output)
    write_json(output / 'execution.json', {'elapsed_seconds': time.monotonic() - started,
               'execution': 'COMPLETED', 'economic_acceptance': 'NOT_PASSED'})
    hashes = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'MANIFEST.json', hashes)
    archive = output.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname='evidence')
    write_json(output.with_suffix('.receipt.json'), {'archive_sha256': file_hash(archive),
               'source_commit': commit, 'report_sha256': file_hash(output / 'report.json'),
               'economic_acceptance': 'NOT_PASSED'})
    print('Research execution complete; economic dominance NOT accepted.', flush=True)


if __name__ == '__main__':
    main()
