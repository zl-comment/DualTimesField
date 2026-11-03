"""Sequential, resumable R18 replay. Running the CLI requires --execute.

Forecast bundles are committed atomically before forecasts enter the ledger.
Resume verifies every bundle and replays ledger updates idempotently. Evaluation
labels are read only after all requested predictions have been committed; earlier
outcomes enter the agent only through the repository's maturity-checked broker.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path('/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012')
CHRONOS_SNAPSHOT = Path('/root/workspace/Code/DualTimesField/weights/hf_home/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c')
CHRONOS_SHA = 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'
CHRONOS_CONFIG_SHA = 'ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031'
GPU_UUID = 'GPU-c931f386-8fcf-9f27-2526-47e12aac3145'
KEV_ROOT = Path('/root/workspace/services/kev')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic JSON replacement on the same filesystem."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def source_identity(root=ROOT):
    paths = sorted((root / 'forecasting').rglob('*.py'))
    for directory in ('skills', 'skills_r19', 'skills_r20'):
        paths += sorted(p for p in (root / 'forecasting/adaptive_agent' / directory).rglob('*') if p.is_file())
    files = {str(path.relative_to(root)): sha_file(path) for path in paths}
    return {'sha256': digest(files), 'files': files}


def policy_module(policy):
    modules = {'r18': '.controller', 'r19_evidence_first': '.evidence_policy',
               'r20_iterative_input': '.iterative_policy'}
    if policy not in modules:
        raise ValueError('Unregistered replay policy')
    return importlib.import_module(modules[policy], package=__package__)


def policy_host_factory(policy):
    if policy == 'r20_iterative_input':
        return importlib.import_module('.iterative_runtime', package=__package__).IterativeToolHost
    if policy in ('r18', 'r19_evidence_first'):
        return importlib.import_module('.runtime', package=__package__).ToolHost
    raise ValueError('Unregistered replay host policy')


def select_indices(length, count):
    if count == 'all':
        return list(range(length))
    count = int(count)
    if not 1 <= count <= length:
        raise ValueError('Origins per region must be positive and no greater than available origins')
    return np.unique(np.linspace(0, length - 1, count, dtype=int)).tolist()


def validate_manifest(output, manifest, *, resume=False):
    """Refuse accidental overwrites and any changed release/input/configuration."""
    output = Path(output)
    manifest_path = output / 'run_manifest.json'
    if resume:
        if not manifest_path.is_file():
            raise ValueError('Resume requires an existing run_manifest.json')
        saved = json.loads(manifest_path.read_text())
        if saved != manifest:
            raise ValueError('Resume identity mismatch: source/configuration/model/input/runtime changed')
    else:
        if output.exists():
            raise ValueError('Output already exists; use a new path or explicit --resume')
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)
    (output / 'origins').mkdir(exist_ok=True)
    return output


def preserve_run_git_provenance(output, current_manifest):
    """Permit provenance-only commits while retaining the frozen run identity.

    All executable source, skills, inputs, models, configuration and package
    identities must still match exactly. Current Git state is audited separately.
    """
    saved = json.loads((Path(output) / 'run_manifest.json').read_text())
    previous_core = {key: value for key, value in saved.items() if key != 'git'}
    current_core = {key: value for key, value in current_manifest.items() if key != 'git'}
    if previous_core != current_core:
        raise ValueError('Resume identity mismatch outside Git provenance')
    result = dict(current_manifest)
    if 'git' in saved:
        result['git'] = saved['git']
    else:
        result.pop('git', None)
    return result


def bundle_name(region, origin):
    return f'{int(origin):012d}_{region}'


def _artifact_record(artifact):
    from .contracts import plain
    return {'candidate_id': artifact.candidate_id, 'context_id': artifact.context_id,
            'model_id': artifact.model_id, 'input_hash': artifact.input_hash,
            'provenance': plain(artifact.provenance), 'identity': artifact.identity()}


def save_bundle(output, context, index, base, agent, trace, audit, manifest_id, *, materialized=None):
    from .contracts import plain
    destination = Path(output) / 'origins' / bundle_name(context.region, context.origin_unix)
    if destination.exists():
        raise ValueError('Cannot overwrite a committed origin')
    temporary = destination.with_name('.pending-' + destination.name + '-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        with (temporary / 'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, base=base.quantiles, agent=agent.quantiles,
                                target_time_unix=context.snapshot.target_time_unix)
            stream.flush()
            os.fsync(stream.fileno())
        write_json(temporary / 'trace.json', plain(trace))
        write_json(temporary / 'audit.json', plain(audit))
        names = ['predictions.npz', 'trace.json', 'audit.json']
        if materialized is not None:
            from .review import validate_artifact
            candidates = [validate_artifact(context, artifact) for artifact in materialized.values()]
            if not candidates or len({item.candidate_id for item in candidates}) != len(candidates):
                raise ValueError('Materialized candidate archive must be nonempty and unique')
            with (temporary / 'materialized_candidates.npz').open('xb') as stream:
                np.savez_compressed(stream,
                    candidate_ids=np.asarray([item.candidate_id for item in candidates]),
                    forecast_ids=np.asarray([item.identity() for item in candidates]),
                    quantiles=np.stack([item.quantiles for item in candidates]),
                    target_time_unix=context.snapshot.target_time_unix)
                stream.flush()
                os.fsync(stream.fileno())
            names.append('materialized_candidates.npz')
        record = {'manifest_id': manifest_id, 'region': context.region,
                  'origin_unix': context.origin_unix, 'index': int(index),
                  'context_id': context.identity(), 'base': _artifact_record(base),
                  'agent': _artifact_record(agent),
                  'files': {name: sha_file(temporary / name) for name in names}}
        write_json(temporary / 'record.json', record)
        os.rename(temporary, destination)
    except BaseException:
        # A process kill can leave .pending directories. They never count as
        # committed and a resume removes only these dedicated staging paths.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_bundle(directory, context, index, manifest_id):
    from .contracts import ForecastArtifact
    directory = Path(directory)
    record = json.loads((directory / 'record.json').read_text())
    if (record['manifest_id'] != manifest_id or record['context_id'] != context.identity()
            or record['region'] != context.region or record['origin_unix'] != context.origin_unix
            or record['index'] != index):
        raise ValueError('Saved origin does not match this run/context')
    required = {'predictions.npz', 'trace.json', 'audit.json'}
    if set(record['files']) not in (required, required | {'materialized_candidates.npz'}):
        raise ValueError('Invalid committed bundle file manifest')
    for name, expected in record['files'].items():
        if sha_file(directory / name) != expected:
            raise ValueError('Committed prediction/trace/audit checksum changed')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'base', 'agent', 'target_time_unix'}:
            raise ValueError('Unexpected prediction payload')
        if not np.array_equal(saved['target_time_unix'], context.snapshot.target_time_unix):
            raise ValueError('Saved target grid changed')
        artifacts = []
        for kind in ('base', 'agent'):
            entry = record[kind]
            artifact = ForecastArtifact(entry['candidate_id'], context.identity(), saved[kind],
                                        saved['target_time_unix'], entry['model_id'],
                                        entry['input_hash'], entry['provenance'])
            if artifact.identity() != entry['identity']:
                raise ValueError('Saved immutable forecast identity changed')
            artifacts.append(artifact)
    trace = json.loads((directory / 'trace.json').read_text())
    if trace.get('selected_forecast_id') != artifacts[1].identity():
        raise ValueError('Trace and selected prediction disagree')
    return artifacts[0], artifacts[1], trace


def build_schedule(repositories, count):
    schedule = [(int(repo.origins[index]), region, int(index))
                for region, repo in repositories.items()
                for index in select_indices(len(repo.origins), count)]
    return sorted(schedule)


def run_replay(output, manifest, repositories, registries, client, evidence, limits,
               *, information_scope='full_evidence', agent_runner=None, host_factory=None):
    """Dependency-injected core shared by real replay and fault/resume tests."""
    from .analogs import AnalogStore
    from .memory import FeedbackLedger
    policy = manifest['configuration'].get('policy', 'r18')
    if agent_runner is None:
        agent_runner = policy_module(policy).run
    host_factory = host_factory or policy_host_factory(policy)
    output = Path(output)
    schedule = build_schedule(repositories, manifest['configuration']['origins_per_region'])
    manifest_id = digest(manifest)
    expected = {bundle_name(region, origin) for origin, region, index in schedule}
    for path in (output / 'origins').iterdir():
        if path.name.startswith('.pending-') and path.is_dir():
            shutil.rmtree(path)
        elif path.name not in expected or not path.is_dir():
            raise ValueError('Unexpected origin entry in resume directory')
    seen_gap = False
    for origin, region, index in schedule:
        exists = (output / 'origins' / bundle_name(region, origin)).is_dir()
        if exists and seen_gap:
            raise ValueError('Committed origins are not a chronological prefix')
        seen_gap |= not exists
    ledger = FeedbackLedger('R18_AGENT:' + manifest_id, output / 'feedback.jsonl')
    # All pending work comes from committed predictions. Idempotent ledger
    # methods repair a crash after bundle rename but before forecast append.
    pending, finished, traces = [], [], []
    initial_reads = {r: len(repo.outcome_reads) for r, repo in repositories.items()}
    started = time.monotonic()
    try:
        for origin, region, index in schedule:
            repo, registry = repositories[region], registries[region]
            context = repo.context(index)
            for item in list(pending):
                if item['available_at'] <= context.cutoff:
                    y = repositories[item['region']].mature_actual(item['index'], context.cutoff)
                    ledger.append_outcome(item['forecast_id'], y, item['available_at'])
                    pending.remove(item)
            directory = output / 'origins' / bundle_name(region, origin)
            if directory.exists():
                base, agent, trace = load_bundle(directory, context, index, manifest_id)
            else:
                episodes = repo.episodes(context) if information_scope == 'full_evidence' else []
                host = host_factory(registry, evidence, AnalogStore(episodes), ledger,
                                    information_scope=information_scope)
                agent, trace = agent_runner(context, host, client, limits)
                base = host.artifact(context, registry.default_id)
                audit = {'tool_audit': host.audit,
                         'outcome_reads': repo.outcome_reads[initial_reads[region]:],
                         'registry_stats': dict(registry.stats),
                         'feedback_before_publish': ledger.fingerprint}
                initial_reads[region] = len(repo.outcome_reads)
                materialized = getattr(host, 'artifacts', None) if policy == 'r20_iterative_input' else None
                save_bundle(output, context, index, base, agent, trace, audit, manifest_id,
                            materialized=materialized)
            key = ledger.append_forecast(context, agent)
            pending.append({'region': region, 'index': index, 'forecast_id': key,
                            'available_at': origin + 86400})
            finished.append({'region': region, 'index': index, 'origin_unix': origin,
                             'base': base, 'agent': agent})
            traces.append(trace)
            write_json(output / 'progress.json', {'status': 'PREDICTING', 'completed': len(finished),
                       'total': len(schedule), 'last_origin': origin, 'last_region': region,
                       'ledger_fingerprint': ledger.fingerprint,
                       'elapsed_this_invocation_seconds': time.monotonic() - started})
            print(json.dumps({'completed': len(finished), 'total': len(schedule),
                              'region': region, 'origin_unix': origin,
                              'selected': agent.candidate_id}), flush=True)
    except BaseException as exc:
        write_json(output / 'progress.json', {'status': 'FAILED', 'completed': len(finished),
                   'total': len(schedule), 'error': type(exc).__name__ + ': ' + str(exc)[:500],
                   'incomplete_origins_in_denominator': len(schedule) - len(finished)})
        raise
    write_json(output / 'predictions_complete.json', {'manifest_id': manifest_id,
               'completed': len(finished), 'expected': len(schedule),
               'ledger_fingerprint': ledger.fingerprint, 'evaluation_labels_opened': False})
    # Dedicated post-prediction phase: evaluation_actual is not reachable by the
    # controller or tools, and cannot execute after an incomplete replay.
    try:
        return _score_completed(output, manifest, repositories, registries, evidence,
                                information_scope, finished, traces, schedule, ledger, manifest_id)
    except BaseException as exc:
        failure = {'status': 'EVALUATION_FAILED', 'manifest_id': manifest_id,
                   'completed_predictions': len(finished), 'expected_origins': len(schedule),
                   'metrics_valid': False,
                   'error': type(exc).__name__ + ': ' + str(exc)[:500]}
        write_json(output / 'progress.json', failure)
        write_json(output / 'report.json', failure)
        raise


def _score_completed(output, manifest, repositories, registries, evidence,
                     information_scope, finished, traces, schedule, ledger, manifest_id):
    reports = {}
    for region, repo in repositories.items():
        expected_actual = manifest.get('inputs', {}).get(region, {}).get('declared_actual_sha256')
        if expected_actual is not None and sha_file(repo.directory / 'actual.npy') != expected_actual:
            raise ValueError('Evaluation outcome file changed relative to frozen cache manifest')
        rows = [row for row in finished if row['region'] == region]
        actual = repo.evaluation_actual([row['index'] for row in rows])
        reports[region] = {'count': len(rows)}
        for kind in ('base', 'agent'):
            predictions = np.stack([row[kind].quantiles for row in rows])
            reports[region][kind.upper()] = aggregate_scores(actual, predictions)
    summary = {'status': 'COMPLETE', 'manifest_id': manifest_id, 'origins': len(finished),
               'expected_origins': len(schedule), 'coverage_fraction': len(finished) / len(schedule),
               'regions': reports, 'region_equal_mean': {},
               'fallback_count': sum(bool(t.get('fallback')) for t in traces),
               'api_calls': sum(t.get('costs', {}).get('api_calls', 0) for t in traces),
               'tools': {}, 'selected_candidates': {},
               'llm_selected_tools': 0, 'protocol_required_tools': 0,
               'unclassified_tools': 0,
               'selection_equals_default': sum(r['agent'].candidate_id == r['base'].candidate_id for r in finished),
               'prediction_differs_from_base': sum(not np.array_equal(r['agent'].quantiles, r['base'].quantiles) for r in finished),
               'failed_origins': 0,
               'abstention_count': sum(t.get('stop_reason') == 'abstain' for t in traces),
               'ledger_fingerprint': ledger.fingerprint,
               'information_scope': information_scope,
               'evidence_fingerprint': evidence.fingerprint}
    latencies = [float(t['elapsed_seconds']) for t in traces if 'elapsed_seconds' in t]
    summary['origin_latency_seconds'] = ({'count': len(latencies), **{
        f'p{p}': float(np.percentile(latencies, p)) for p in (50, 95, 99)}} if latencies else {'count': 0})
    calls = [call.get('record', call) for trace in traces for call in trace.get('calls', [])]
    summary['api_failure_count'] = sum('error' in record for record in calls)
    api_times = [float(record['elapsed_seconds']) for record in calls if 'elapsed_seconds' in record]
    summary['api_latency_seconds'] = ({'count': len(api_times), **{
        f'p{p}': float(np.percentile(api_times, p)) for p in (50, 95, 99)}} if api_times else {'count': 0})
    summary['serial_full_grid_hours_estimate'] = (
        float(np.mean(latencies) * 157689 / 3600) if latencies else None)
    summary['estimate_limit'] = 'Serial origin latency only; excludes startup, file hashing, and environment changes. Not an accuracy estimate.'
    summary['capabilities'] = {
        'frozen_lightgbm': all('lightgbm' in r.adapters for r in registries.values()),
        'frozen_chronos': all('chronos' in r.adapters for r in registries.values()),
        'historical_analogs': information_scope == 'full_evidence',
        'mature_feedback': information_scope == 'full_evidence',
        'external_evidence': manifest.get('evidence', {}).get('status', 'test_fixture'),
        'notice': 'Enabled capability does not imply selected at every origin; see tool counts.'}
    for kind in ('BASE', 'AGENT'):
        summary['region_equal_mean'][kind] = {key: float(np.mean([r[kind][key] for r in reports.values()]))
                                            for key in next(iter(reports.values()))[kind]}
    for row in finished:
        key = row['agent'].candidate_id
        summary['selected_candidates'][key] = summary['selected_candidates'].get(key, 0) + 1
    for trace in traces:
        for tool in trace.get('tools', []):
            key = tool.get('action', tool.get('action_id', 'unknown'))
            summary['tools'][key] = summary['tools'].get(key, 0) + 1
            mode = tool.get('execution_mode')
            if mode is None and manifest['configuration'].get('policy', 'r18') == 'r18':
                # Legacy R18 tools were all selected by the LLM; its trace
                # predates the explicit protocol-vs-LLM attribution field.
                mode = 'llm_selected'
            field = {'llm_selected': 'llm_selected_tools',
                     'protocol_required': 'protocol_required_tools'}.get(mode, 'unclassified_tools')
            summary[field] += 1
    summary['tool_count_semantics'] = 'Counts trace tool attempts, including recorded errors; protocol-required coverage is not evidence of LLM benefit.'
    summary['policy'] = manifest['configuration'].get('policy', 'r18')
    summary['iteration_count'] = sum(len(trace.get('iterations', [])) for trace in traces)
    summary['revision_count'] = sum(len(trace.get('revisions', [])) for trace in traces)
    summary['origins_with_revision'] = sum(bool(trace.get('revisions')) for trace in traces)
    summary['optional_evidence_action_count'] = sum(len(trace.get('optional_evidence_actions', [])) for trace in traces)
    summary['iteration_semantics'] = 'Candidate comparison rounds and optional LLM revisions are counted from explicit trace lists; missing legacy fields count as zero.'
    summary['origin_trace_costs'] = [{
        'region': row['region'], 'origin_unix': row['origin_unix'],
        'trace_path': 'origins/' + bundle_name(row['region'], row['origin_unix']) + '/trace.json',
        'costs': trace.get('costs', {}),
        'iteration_count': len(trace.get('iterations', [])),
        'revision_count': len(trace.get('revisions', [])),
        'optional_evidence_action_count': len(trace.get('optional_evidence_actions', [])),
        'elapsed_seconds': trace.get('elapsed_seconds'),
        'selected_forecast_id': row['agent'].identity(),
        'fallback': trace.get('fallback')}
        for row, trace in zip(finished, traces)]
    write_json(output / 'report.json', summary)
    write_json(output / 'progress.json', {'status': 'COMPLETE', 'completed': len(finished), 'total': len(schedule)})
    return summary


def aggregate_scores(actual, quantiles):
    levels = np.array([.05, .1, .5, .9, .95])
    actual, q = np.asarray(actual), np.asarray(quantiles)
    if q.shape != (*actual.shape, 5) or not np.isfinite(q).all() or not np.isfinite(actual).all():
        raise ValueError('Evaluation arrays invalid')
    q = np.sort(q, axis=-1)
    error = actual[..., None] - q
    residual = actual - q[..., 2]
    return {'qs': float(2 * np.maximum(levels * error, (levels - 1) * error).mean()),
            'mae': float(np.abs(residual).mean()), 'rmse': float(np.sqrt(np.square(residual).mean())),
            'coverage90': float(((actual >= q[..., 0]) & (actual <= q[..., -1])).mean()),
            'width90': float((q[..., -1] - q[..., 0]).mean())}


def verify_input_files(repository):
    from ..aligned_inputs import INPUT_KEYS
    declarations = json.loads((repository.directory.parent / 'hashes.json').read_text())
    split = repository.directory.name
    names = ['metadata.json', 'origin_unix.npy', 'target_time_unix.npy'] + [f'input__{key}.npy' for key in INPUT_KEYS]
    verified = {}
    for name in names:
        sha = sha_file(repository.directory / name)
        if sha != declarations[f'{split}/{name}']:
            raise ValueError('Input cache checksum mismatch: ' + str(repository.directory / name))
        verified[name] = sha
    return {'directory': str(repository.directory.resolve()), 'files': verified,
            'cache_manifest_sha256': sha_file(repository.directory.parent / 'hashes.json'),
            'declared_actual_sha256': declarations[f'{split}/actual.npy'],
            'actual_hash_verification_phase': 'after_all_predictions'}


def verify_kev_files():
    path = KEV_ROOT / 'verified_weight_manifest.json'
    records = json.loads(path.read_text())
    if not records:
        raise ValueError('Missing verified Kev weight manifest')
    for filename, record in records.items():
        file = Path(filename).resolve()
        if not file.is_relative_to((KEV_ROOT / 'models').resolve()):
            raise ValueError('Kev model manifest path outside pinned model directory')
        if file.stat().st_size != record['size'] or sha_file(file) != record['sha256']:
            raise ValueError('Kev runtime checkpoint checksum mismatch')
    return {'manifest_sha256': sha_file(path), 'files': records}


def stable_service_identity(identity):
    """Operational counters must not invalidate a deterministic resume."""
    normalized = json.loads(json.dumps(identity))
    for model in normalized.get('models', []):
        model.pop('batches', None)
        if 'prefix_cache' in model:
            model['prefix_cache'] = {key: value for key, value in model['prefix_cache'].items()
                                     if key in ('size', 'min_state_tokens', 'max_tokens')}
    return normalized


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly execute live model/API inference')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('selection', 'test'), default='selection')
    parser.add_argument('--origins-per-region', '--origins', default='24', help='Positive count or all; evenly spaced existing origins')
    parser.add_argument('--regions', nargs='+', choices=('NSW1', 'QLD1', 'TAS1'), default=['NSW1', 'QLD1', 'TAS1'])
    parser.add_argument('--seed', type=int, choices=(2026, 2027, 2028), default=2026)
    parser.add_argument('--policy', choices=('r18', 'r19_evidence_first', 'r20_iterative_input'), default='r18')
    parser.add_argument('--information-scope', choices=('full_evidence', 'common_input'), default='full_evidence')
    parser.add_argument('--evidence', help='Frozen evidence JSONL; absent means real_data_unavailable')
    parser.add_argument('--api-calls', type=int, default=8)
    parser.add_argument('--origin-deadline-seconds', type=float, default=180)
    parser.add_argument('--baseline-root', default=str(BASELINE_ROOT))
    args = parser.parse_args(argv)
    if len(set(args.regions)) != len(args.regions):
        parser.error('Duplicate regions')
    configuration = {'split': args.split, 'origins_per_region': args.origins_per_region,
                     'policy': args.policy,
                     'regions': sorted(args.regions), 'seed': args.seed,
                     'information_scope': args.information_scope,
                     'api_calls': args.api_calls, 'origin_deadline_seconds': args.origin_deadline_seconds,
                     'gpu_uuid': GPU_UUID, 'kev_url': 'http://127.0.0.1:8009/v1/systemone',
                     'baseline_root': str(Path(args.baseline_root).resolve()),
                     'evidence_path': str(Path(args.evidence).resolve()) if args.evidence else None}
    if not args.execute:
        print(json.dumps({'status': 'PLAN_ONLY', 'configuration': configuration,
                          'instruction': 'Add --execute to run; no model/API/output modification performed.'}, indent=2))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    from .contracts import BudgetLimits
    from .context import ContextRepository
    selected_policy = policy_module(args.policy)
    release, load_skills = selected_policy.RELEASE, selected_policy.load_skills
    from .evidence import EvidenceStore
    from .registry import ForecastRegistry
    from .predictors.chronos import FrozenChronosAdapter
    from .predictors.lightgbm import FrozenLightGBMAdapter
    from ..api_agent.transport import KevClient
    from chronos import Chronos2Pipeline
    limits = BudgetLimits(api_calls=args.api_calls, origin_deadline_seconds=args.origin_deadline_seconds)
    base_root = Path(args.baseline_root)
    repositories = {region: ContextRepository(base_root / 'data_cache' / region / args.split)
                    for region in configuration['regions']}
    schedule = build_schedule(repositories, args.origins_per_region)
    inputs = {region: verify_input_files(repo) for region, repo in repositories.items()}
    if sha_file(CHRONOS_SNAPSHOT / 'model.safetensors') != CHRONOS_SHA or sha_file(CHRONOS_SNAPSHOT / 'config.json') != CHRONOS_CONFIG_SHA:
        raise ValueError('Pinned Chronos weights/config checksum mismatch')
    kev_identity = verify_kev_files()
    client = KevClient(timeout=limits.api_timeout_seconds)
    kev_identity['service'] = stable_service_identity(client.identity())
    kev_identity['serve_script_sha256'] = sha_file(KEV_ROOT / 'serve.sh')
    kev_identity['requirements_sha256'] = sha_file(KEV_ROOT / 'requirements.freeze.txt')
    evidence = EvidenceStore.from_jsonl(args.evidence) if args.evidence else EvidenceStore()
    chronos_identity = {'sha256': CHRONOS_SHA, 'config_sha256': CHRONOS_CONFIG_SHA,
                        'revision': CHRONOS_SNAPSHOT.name, 'path': str(CHRONOS_SNAPSHOT),
                        'pretraining_overlap': 'unknown'}
    adapters = {}
    for region in repositories:
        adapters[region] = FrozenLightGBMAdapter(base_root / 'runs/lightgbm' / f'seed{args.seed}' / region,
                                                region=region, seed=args.seed, threads=4)
    manifest = {'schema': 'r18-sequential-replay-v1', 'release': release,
                'configuration': configuration, 'budgets': asdict(limits), 'source': source_identity(),
                'inputs': inputs, 'models': {'chronos': chronos_identity, 'kev': kev_identity,
                'lightgbm': {region: adapter.identity for region, adapter in adapters.items()}},
                'skills': {name: {'version': value['version'], 'sha256': value['sha256']}
                           for name, value in load_skills().items()},
                'evidence': {'fingerprint': evidence.fingerprint,
                             'file_sha256': sha_file(args.evidence) if args.evidence else None,
                             'status': 'configured_frozen_warehouse' if args.evidence else 'real_data_unavailable'},
                'packages': {name: importlib.metadata.version(name) for name in
                             ('numpy', 'torch', 'chronos-forecasting', 'lightgbm', 'transformers')},
                'schedule_sha256': digest(schedule), 'expected_origins': len(schedule),
                'scoring_rule': 'Only after every requested prediction has been atomically committed.',
                'seed_semantics': 'LightGBM checkpoint seed; Kev argmax does not sample a random seed.',
                'historical_evaluation_limit': 'Selection/test exposure retained; no independent-blind-test claim.'}
    manifest['git'] = {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'worktree': str(ROOT),
        'tracked_changes': subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines(),
        'uncommitted_source_bound_by_manifest': True}
    invocation_git = manifest['git']
    if args.resume:
        manifest = preserve_run_git_provenance(args.output, manifest)
    output = validate_manifest(args.output, manifest, resume=args.resume)
    # Serialize concurrent resumptions before touching any origin or ledger.
    import fcntl
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        invocations = output / 'invocations'
        invocations.mkdir(exist_ok=True)
        write_json(invocations / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json'),
                   {'current_git': invocation_git, 'resume': args.resume,
                    'manifest_id': digest(manifest), 'configuration': configuration})
        pipeline = Chronos2Pipeline.from_pretrained(str(CHRONOS_SNAPSHOT), device_map='cuda:0', local_files_only=True)
        chronos = FrozenChronosAdapter(pipeline, chronos_identity)
        registries = {region: ForecastRegistry({'lightgbm': adapter, 'chronos': chronos},
                                               release, max_current_candidates=limits.candidates)
                      for region, adapter in adapters.items()}
        report = run_replay(output, manifest, repositories, registries, client, evidence, limits,
                            information_scope=args.information_scope, agent_runner=selected_policy.run)
        if source_identity() != manifest['source']:
            write_json(output / 'source_changed.json', {'status': 'INVALIDATED', 'reason': 'Source changed during replay'})
            report['status'] = 'INVALIDATED'
            write_json(output / 'report.json', report)
            write_json(output / 'progress.json', {'status': 'INVALIDATED', 'reason': 'Source changed during replay'})
            raise ValueError('Source changed during run; results marked invalidated')
        print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
