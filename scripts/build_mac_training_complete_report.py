"""Collect Mac evidence and build an executed, portable training report.

No TensorFlow import, training, checkpoint modification or live hardware query.
Raw outputs take precedence; compact historical snapshots fill missing runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "analysis_reports/mac_training_complete_2026-09-30"
DATA = REPORT / "evidence"
NOTEBOOK = PROJECT / "notebooks/mac_all_training_complete_report.ipynb"
SNAPSHOT = PROJECT / "analysis_reports/activation_quantization_2026-09-30/data"
METRICS = {
    "cpu": "cpu_percent", "process_cpu": "process_cpu_percent",
    "gpu": "gpu_utilization_percent", "ram": "ram_percent",
    "gpu_memory": "gpu_memory_used_bytes", "process_rss": "process_rss_bytes",
}
EPOCH_FIELDS = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy",
                "val_balanced_accuracy", "val_macro_f1", "learning_rate",
                "epoch_seconds", "train_examples", "train_examples_per_second"]


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def collect():
    DATA.mkdir(parents=True, exist_ok=True)
    rows, epochs, classes, timelines, environments, checks, sources = [], [], [], [], [], [], []
    excluded = []

    def source(path):
        if not path.is_file():
            return
        rel = str(path.relative_to(PROJECT))
        if rel in {x['path'] for x in sources}:
            return
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        sources.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest.hexdigest()))

    for status_path in sorted((PROJECT / 'outputs').rglob('status.json')):
        run = status_path.parent
        env_path = run / 'telemetry/environment.json'
        manifest_path = run / 'manifest.json'
        env, manifest, status = read_json(env_path), read_json(manifest_path), read_json(status_path)
        if not manifest:
            continue  # Partial mirrored summaries are reconciled against the snapshot below.
        config = manifest.get('config', {})
        if config.get('training', {}).get('extra_fraction') != 0.5:
            continue
        is_mac = env.get('platform', {}).get('system') == 'Darwin'
        inferred_mac = not env and '/Users/' in str(config.get('dataset_root', ''))
        if not (is_mac or inferred_mac):
            excluded.append(dict(source_ref=str(status_path.relative_to(PROJECT)),
                                 status=status.get('status'),
                                 platform=env.get('platform', {}).get('system', 'unknown'),
                                 gpu=(env.get('hardware', {}).get('gpus') or [{}])[0].get('name'),
                                 reason='Ambiente não é macOS; presença de arquivo no Mac não indica origem.'))
            continue
        rel = status_path.relative_to(PROJECT)
        campaign, stage = rel.parts[1:3]
        train = config.get('training', {})
        protocol = config.get('protocol', {})
        condition = 'smoke' if stage == 'smoke' else f"batch-{train.get('batch_size')}"
        if stage == 'activations':
            condition = train.get('hidden_activation')
        elif stage == 'quantization':
            condition = next((p for p in rel.parts if p in ['fp16', 'fp32', 'int8_qat', 'int4_qat', 'int8_ptq']), 'unknown')
        uid = '|'.join([campaign, stage, config.get('dataset', ''), condition])
        summary = read_json(run / 'logs/training_summary.json')
        test = read_json(run / 'artifacts/test_metrics.json')
        classification = test.get('classification', {})
        telem = read_json(run / 'telemetry/summary.json')
        hardware = env.get('hardware', {})
        gpu = (hardware.get('gpus') or [{}])[0]
        row = dict(uid=uid, campaign=campaign, stage=stage, dataset=config.get('dataset'), condition=condition,
                   run_id=manifest.get('run_id'), status=status.get('status'),
                   status_updated_at=status.get('status_updated_at'), attempt=manifest.get('attempt'),
                   source_ref=str(rel), source_kind='raw', environment_basis='environment.json' if is_mac else 'Mac path in manifest (planned)',
                   batch_size=train.get('batch_size'), extra_fraction=train.get('extra_fraction'),
                   seed=config.get('seed'), normalization=config.get('normalization'), balance_mode=config.get('balance_mode'),
                   dtype_policy=train.get('dtype_policy'), hidden_activation=train.get('hidden_activation'),
                   learning_rate=train.get('learning_rate'), num_classes=config.get('num_classes'),
                   image_size=config.get('target_size'), native_channels=config.get('native_channels'),
                   train_fraction=config.get('train_fraction'), validation_fraction=config.get('validation_fraction'),
                   test_fraction=config.get('test_fraction'), max_epochs=train.get('max_epochs'),
                   config_fingerprint=manifest.get('config_fingerprint'), split_fingerprint=manifest.get('split_fingerprint'),
                   source_fingerprint=config.get('source_fingerprint'),
                   optimizer=protocol.get('optimizer'), epochs_completed=summary.get('epochs_completed'),
                   training_seconds_reported=summary.get('training_seconds', summary.get('total_seconds')),
                   evaluation_seconds=summary.get('evaluation_seconds'), wall_seconds=summary.get('wall_seconds_current_attempt'),
                   gpu_backend=hardware.get('gpu_backend'), gpu_name=gpu.get('name'),
                   gpu_core_count=gpu.get('gpu_core_count'), system_ram_gib=hardware.get('ram_total_bytes', np.nan) / 1024**3,
                   cpu_logical_cores=hardware.get('cpu_count_logical'),
                   macos_version=hardware.get('macos_version'), metal_version=hardware.get('metal_version'),
                   tensorflow_version=env.get('packages', {}).get('tensorflow'),
                   tensorflow_metal_version=env.get('packages', {}).get('tensorflow_metal'),
                   python_version=env.get('python', {}).get('version'),
                   test_loss=test.get('keras_metrics', {}).get('loss'),
                   telemetry_samples=telem.get('sample_count'), telemetry_interval_seconds=telem.get('interval_seconds'),
                   thermal_pressure=telem.get('categorical', {}).get('thermal_pressure', {}).get('current'))
        for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'macro_ovr_auc']:
            row['test_' + key] = classification.get(key)
        row['test_samples'] = sum(c.get('support', 0) for c in classification.get('per_class', {}).values()) or None
        for key, c in classification.get('per_class', {}).items():
            classes.append(dict(uid=uid, class_label=key, class_name=(config.get('class_names') or [])[int(key)]
                                if str(key).isdigit() and int(key) < len(config.get('class_names', [])) else key,
                                **{k: c.get(k) for k in ['f1', 'precision', 'recall', 'support']}))
        epoch_path = run / 'checkpoints/epoch_metrics.csv'
        if not epoch_path.exists():
            epoch_path = run / 'logs/epoch_metrics.csv'
        if epoch_path.is_file():
            history = pd.read_csv(epoch_path)
            duplicates = history.epoch.duplicated().sum()
            checks.append(dict(uid=uid, check='unique_epoch', passed=not bool(duplicates), detail=f'{duplicates} duplicadas; usa última por época'))
            history = history.drop_duplicates('epoch', keep='last').sort_values('epoch')
            history = history.reindex(columns=EPOCH_FIELDS)
            history['uid'] = uid
            epochs.extend(history.to_dict('records'))
            row['epochs_recorded'] = len(history)
            row['epochs_completed'] = len(history) if not summary else summary.get('epochs_completed')
        for prefix, name in METRICS.items():
            for stat in ['mean', 'p95', 'max']:
                val = telem.get('metrics', {}).get(name, {}).get(stat)
                field = f'{prefix}_{stat}_' + ('gib' if prefix in ['gpu_memory', 'process_rss'] else 'percent')
                row[field] = val / 1024**3 if val is not None and prefix in ['gpu_memory', 'process_rss'] else val
        samples_path = run / 'telemetry/samples.csv'
        row['hardware_scope'] = 'Resumo registrado da execução; inclui preparação/avaliação e eventos'
        if samples_path.is_file():
            cols = ['timestamp', 'elapsed_seconds', 'event', 'thermal_pressure', *METRICS.values()]
            samples = pd.read_csv(samples_path, usecols=lambda c: c in cols, low_memory=False)
            stamps = pd.to_datetime(samples.timestamp, utc=True)
            # Elapsed time can restart when resuming. Bin by UTC, never merge different attempts at the same elapsed value.
            samples['session'] = (samples.elapsed_seconds.diff().lt(0) | samples.event.eq('train_start')).cumsum()
            row['telemetry_sessions'] = int(samples.session.nunique())
            periodic = samples.loc[samples.event.eq('periodic')].copy()
            stats_frame = periodic if len(periodic) else samples
            row['hardware_scope'] = 'Todas as amostras periódicas preservadas no CSV (ciclo inteiro, todas as sessões disponíveis)'
            row['telemetry_samples_raw'] = len(samples)
            row['telemetry_periodic_samples'] = len(periodic)
            row['telemetry_start'] = str(stamps.min())
            row['telemetry_end'] = str(stamps.max())
            for prefix, name in METRICS.items():
                values = pd.to_numeric(stats_frame[name], errors='coerce')
                if prefix in ['gpu_memory', 'process_rss']:
                    values = values / 1024**3
                for stat, val in [('mean', values.mean()), ('p95', values.quantile(.95)), ('max', values.max())]:
                    row[f'{prefix}_{stat}_' + ('gib' if prefix in ['gpu_memory', 'process_rss'] else 'percent')] = val
            samples['minute'] = stamps.dt.floor('min').astype(str)
            samples['offset_seconds'] = (stamps - stamps.min()).dt.total_seconds()
            for name in METRICS.values():
                samples[name] = pd.to_numeric(samples[name], errors='coerce')
            agg = {'offset_seconds': 'mean', **{name: ['mean', 'min', 'max', 'count'] for name in METRICS.values()}}
            binned = samples.groupby(['session', 'minute']).agg(agg)
            binned.columns = ['_'.join(c) for c in binned.columns]
            binned = binned.reset_index()
            binned['uid'] = uid
            timelines.extend(binned.to_dict('records'))
            if stats_frame.thermal_pressure.notna().any():
                row['thermal_pressure'] = ', '.join(sorted(stats_frame.thermal_pressure.dropna().unique()))
        if env:
            environments.append(dict(uid=uid, source_ref=str(env_path.relative_to(PROJECT)), environment=env))
        rows.append(row)
        for p in [status_path, manifest_path, env_path, epoch_path, samples_path,
                  run / 'artifacts/test_metrics.json', run / 'logs/training_summary.json', run / 'telemetry/summary.json']:
            source(p)

    # Exported companion campaigns: raw mirrored summaries are not separate executions.
    snap_runs = pd.read_csv(SNAPSHOT / 'runs.csv')
    snap_epochs = pd.read_csv(SNAPSHOT / 'epoch_history.csv')
    snap_classes = pd.read_csv(SNAPSHOT / 'per_class_metrics.csv')
    for _, sr in snap_runs.iterrows():
        campaign = sr.source_ref.split('/')[1]
        stage = 'ptq' if sr.stage == 'post_training_quantization_evaluation' else ('activations' if sr.campaign == 'activation' else 'quantization')
        uid = '|'.join([campaign, stage, sr.dataset, sr.condition])
        if uid in {r['uid'] for r in rows}:
            continue
        row = sr.to_dict()
        row.update(uid=uid, campaign=campaign, stage=stage, source_kind='exported_snapshot',
                   environment_basis='Export Mac + metadata campaign; no per-run environment here',
                   hardware_scope='Resumo exportado; ciclo da execução, inclui eventos; amostras brutas indisponíveis',
                   training_seconds_reported=sr.training_seconds,
                   hidden_activation=sr.condition if stage == 'activations' else 'swish',
                   source_snapshot=str((SNAPSHOT / 'runs.csv').relative_to(PROJECT)))
        rows.append(row)
        history = snap_epochs.loc[(snap_epochs.campaign == sr.campaign) & (snap_epochs.dataset == sr.dataset) & (snap_epochs.condition == sr.condition)].copy()
        if stage != 'ptq':
            history['uid'] = uid
            epochs.extend(history.reindex(columns=['uid', *EPOCH_FIELDS]).to_dict('records'))
        cs = snap_classes.loc[(snap_classes.campaign == sr.campaign) & (snap_classes.dataset == sr.dataset) & (snap_classes.condition == sr.condition)].copy()
        cs['uid'] = uid
        classes.extend(cs[['uid', 'class_label', 'class_name', 'precision', 'recall', 'f1', 'support']].to_dict('records'))
        # Validate any mirrored local result rather than counting it twice.
        mirror = PROJECT / sr.source_ref
        summary_path = mirror.parent.parent.parent / 'summary.json'
        if summary_path.is_file():
            local = read_json(summary_path).get('runs', [{}])[0]
            checks.append(dict(uid=uid, check='snapshot_vs_local_summary',
                               passed=np.isclose(local.get('test_macro_f1', np.nan), sr.test_macro_f1),
                               detail='Macro-F1 exportado comparado ao summary.json local'))
            source(summary_path)
    for p in SNAPSHOT.glob('*'):
        source(p)
    metadata_path = PROJECT / 'docs/augmentation05_final/data/resumo_augmentation05.json'
    source(metadata_path)
    source(PROJECT / 'src/tcc_benchmark/model.py')
    for p in (PROJECT / 'configs').glob('*mac*.yaml'):
        source(p)
    runs, epoch_df, class_df = pd.DataFrame(rows), pd.DataFrame(epochs), pd.DataFrame(classes)
    assert not runs.uid.duplicated().any(), 'Duplicate execution identities'
    for idx, row in runs.iterrows():
        h = epoch_df.loc[epoch_df.uid.eq(row.uid)]
        runs.loc[idx, 'epochs_recorded'] = len(h)
        runs.loc[idx, 'epochs_timed'] = h.epoch_seconds.notna().sum()
        if len(h):
            full_timing = h.epoch_seconds.notna().all()
            seconds = h.epoch_seconds.sum(min_count=1)
            runs.loc[idx, 'training_seconds'] = seconds if full_timing else np.nan
            runs.loc[idx, 'timing_basis'] = 'Soma das épocas preservadas' if full_timing else 'Histórico com lacunas de duração'
            runs.loc[idx, 'mean_epoch_seconds'] = h.epoch_seconds.mean()
            valid = h.dropna(subset=['train_examples', 'epoch_seconds'])
            runs.loc[idx, 'train_examples_per_second'] = valid.train_examples.sum() / valid.epoch_seconds.sum() if len(valid) else np.nan
            f1 = h.dropna(subset=['val_macro_f1'])
            runs.loc[idx, 'best_epoch'] = f1.loc[f1.val_macro_f1.idxmax(), 'epoch'] if len(f1) else np.nan
            runs.loc[idx, 'best_val_macro_f1'] = f1.val_macro_f1.max()
            runs.loc[idx, 'final_val_macro_f1'] = f1.val_macro_f1.iloc[-1] if len(f1) else np.nan
            reported = row.get('training_seconds_reported')
            if pd.notna(reported):
                delta = seconds - reported
                runs.loc[idx, 'timing_delta_seconds'] = delta
                checks.append(dict(uid=row.uid, check='epoch_sum_vs_reported', passed=abs(delta) < 1,
                                   detail=f'Soma de épocas menos resumo = {delta:.3f}s; conservar ambas as medidas, usar histórico completo'))
            checks.append(dict(uid=row.uid, check='epoch_sequence', passed=h.epoch.tolist() == list(range(1, len(h)+1)),
                               detail=f'{len(h)} épocas únicas'))
            if row.status == 'completed' and pd.notna(row.get('epochs_completed')):
                checks.append(dict(uid=row.uid, check='epoch_count', passed=len(h) == row.epochs_completed,
                                   detail=f'{len(h)} registradas / {row.epochs_completed} declaradas'))
        elif row.source_kind == 'historical_reference':
            runs.loc[idx, 'timing_basis'] = 'Resumo histórico; sem curva para reconciliar'
        cs = class_df.loc[class_df.uid.eq(row.uid)]
        if len(cs):
            for field, computed in [('test_macro_f1', cs.f1.mean()), ('test_accuracy', np.average(cs.recall, weights=cs.support))]:
                checks.append(dict(uid=row.uid, check='classwise_' + field, passed=np.isclose(computed, row[field], atol=1e-7),
                                   detail=f'Recalculado por classe = {computed:.9f}; registro = {row[field]:.9f}'))
    runs['training_hours'] = runs.training_seconds / 3600
    for name, frame in [('runs', runs), ('epoch_history', epoch_df), ('per_class_metrics', class_df),
                        ('telemetry_minute', pd.DataFrame(timelines)), ('quality_checks', pd.DataFrame(checks)),
                        ('excluded_non_mac', pd.DataFrame(excluded))]:
        frame.to_csv(DATA / f'{name}.csv', index=False, float_format='%.9g')
    write_json(DATA / 'environments.json', environments)
    write_json(DATA / 'campaign_metadata.json', read_json(metadata_path))
    write_json(DATA / 'sources.json', sources)
    manifest = dict(report_date_local='2026-09-30', timezone='America/Sao_Paulo',
                    scope='Apenas Mac com extra_fraction=0.5, nas saídas e snapshots do projeto. INT8 PTQ vinculado à campanha; smoke em apêndice.',
                    record_count=len(runs), epochs=len(epoch_df), classes=len(class_df),
                    timeline_bins=len(timelines), excluded_non_mac=len(excluded),
                    status_counts=runs.groupby(['campaign', 'stage', 'status']).size().to_dict(),
                    telemetry_bin_seconds=60, sources_count=len(sources))
    manifest['status_counts'] = {' | '.join(k): v for k, v in manifest['status_counts'].items()}
    write_json(DATA / 'manifest.json', manifest)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Execute notebook and render HTML with this Python environment')
    args = parser.parse_args()
    runs = collect()
    print(runs.groupby(['campaign', 'stage', 'status']).size().to_string())
    if not args.collect_only:
        from mac_report_notebook import build_notebook
        build_notebook(PROJECT, REPORT, NOTEBOOK, runs)
        if args.execute:
            execute_notebook()


def execute_notebook():
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    notebook = nbf.read(NOTEBOOK, as_version=4)
    with tempfile.TemporaryDirectory(prefix='mac-report-kernel-') as folder:
        kernel_path = Path(folder) / 'python3'
        kernel_path.mkdir()
        write_json(kernel_path / 'kernel.json', dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                                    display_name='Python 3 (relatório Mac)', language='python'))
        manager = KernelManager(kernel_name='python3', transport='ipc',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[folder]))
        try:
            NotebookClient(notebook, timeout=180, km=manager, resources={'metadata': {'path': str(PROJECT)}}).execute()
        finally:
            nbf.write(notebook, NOTEBOOK)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbf.validate(notebook)
    html, _ = HTMLExporter(template_name='lab', exclude_input=True).from_notebook_node(notebook)
    (REPORT / 'mac_training_complete_report.html').write_text(html)
    print(f'Executed notebook and HTML: {NOTEBOOK}')


if __name__ == '__main__':
    main()
