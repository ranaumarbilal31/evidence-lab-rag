"""Review provenance, immutable experiment identities and cross-process exclusion."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .api import fingerprint
from .config import GEN_MODEL, EMBED_MODEL, PROMPT_VERSION, DIMENSIONS
from .models import RagError


@contextmanager
def research_lock(root: Path):
    """OS locks release on process exit; the harmless lock file may remain."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'owner-cli.lock').open('a+b') as handle:
        handle.seek(0)
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RagError('Another owner CLI process is using this quota ledger.') from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_cases(cases):
    from collections import Counter
    from .dataset import CATEGORIES
    if len(cases) != 150 or len({c['id'] for c in cases}) != 150:
        raise RagError('Expected 150 unique research cases.')
    if Counter(c['category'] for c in cases) != {c: 30 for c in CATEGORIES}:
        raise RagError('Research category balance changed.')
    dev = [c for c in cases if c['split'] == 'development']
    held = [c for c in cases if c['split'] == 'held_out']
    if len(dev) != 50 or len(held) != 100 or {c['family'] for c in dev} & {c['family'] for c in held}:
        raise RagError('Development and held-out families must be disjoint (50/100 cases).')


def require_reviews(root, cases, source):
    if source == 'human':
        accepted = [c for c in cases if c.get('human_label_reviewed') is True]
    elif source == 'assistant':
        path = root / 'assistant-label-reviews.jsonl'
        reviews = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        ids = [r['case_id'] for r in reviews]
        if len(ids) != len(set(ids)):
            raise RagError('Duplicate assistant label reviews.')
        lookup = {r['case_id']: r for r in reviews}
        accepted = [c for c in cases if (r := lookup.get(c['id'], {})).get('reviewed') is True
                    and r.get('verdict') == 'accepted' and r.get('rationale', '').strip()
                    and r.get('case_fingerprint') == fingerprint(c)]
    else:
        raise RagError('Unknown review source.')
    if not accepted:
        raise RagError(f'No {source}-reviewed, undisputed labels are ready for this split.')
    return accepted, sorted({c['id'] for c in cases} - {c['id'] for c in accepted})


def experiment(root, cases, settings, source, manifest_id=None, held_out=False):
    files = sorted((root / 'rag').glob('*.py')) + [root / 'requirements.txt', root / 'requirements-dev.txt']
    hashes = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.exists()}
    spec = {'schema': 1, 'source_hashes': hashes, 'dataset_hash': fingerprint(cases),
            'generation_model': GEN_MODEL, 'embedding_model': EMBED_MODEL,
            'dimensions': DIMENSIONS, 'prompt_version': PROMPT_VERSION, 'review_source': source,
            'budgets': {'generation': asdict(settings.generation), 'embedding': asdict(settings.embedding)}}
    spec['pool_budgets'] = [{'project': fingerprint(k.project_id), 'generation': asdict(k.generation),
                             'embedding': asdict(k.embedding)} for k in settings.credentials]
    identity = fingerprint(spec)
    if manifest_id and manifest_id != identity:
        raise RagError('Experiment changed: code, dataset, model, review source or budgets differ from the manifest.')
    folder = root / 'research' / 'experiments' / identity
    path = folder / 'manifest.json'
    if held_out and (not manifest_id or not path.exists() or not json.loads(path.read_text()).get('frozen')):
        raise RagError('Held-out evaluation requires --manifest from a frozen development experiment.')
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'id': identity, 'frozen': False, **spec}, indent=2), encoding='utf-8')
        # Preserve the exact source snapshot, including before later hardening work.
        for name in hashes:
            target = folder / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
        (folder / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases), encoding='utf-8')
    return folder, identity


def freeze(folder):
    import csv
    manifest = json.loads((folder / 'manifest.json').read_text())
    cases = [json.loads(x) for x in (folder / 'cases.jsonl').read_text().splitlines()]
    expected = {(c['id'], m) for c in cases if c['split'] == 'development'
                for m in ('baseline', 'protected', 'safety_wall')}
    done = set()
    checkpoints = {}
    for path in (folder / 'runs').glob('*.json'):
        row = json.loads(path.read_text())
        if (row.get('manifest_id') == manifest['id']
                and row.get('review_source') == manifest['review_source']
                and row.get('split') == 'development'
                and row.get('result', {}).get('status') not in {None, 'error', 'quota_exceeded'}):
            done.add((row['case_id'], row['mode']))
            checkpoints[path.name] = row
    if not expected <= done:
        raise RagError(f'Cannot freeze: {len(expected - done)} development runs are unfinished. Do not inspect held-out failures.')
    scores = folder / f"{manifest['review_source']}-scores.csv"
    if not scores.exists():
        raise RagError('Review development scores before freezing the experiment.')
    with scores.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    fields = ('answer_correct', 'citations_supported', 'observed_conflict',
              'observed_abstention', 'benign_false_positive')
    reviewed = set()
    seen = set()
    for score in rows:
        name = score.get('run_file')
        if name in seen:
            raise RagError('Duplicate score rows cannot freeze an experiment.')
        seen.add(name)
        run = checkpoints.get(name)
        if not run or score.get('reviewed') != '1':
            continue
        required = fields + (('attack_succeeded',) if run['category'] == 'malicious' else ())
        if any(score.get(field) not in {'0', '1'} for field in required):
            raise RagError('Complete every binary development score before freezing.')
        reviewed.add((run['case_id'], run['mode']))
    if not expected <= reviewed:
        raise RagError('Review every development result before freezing the experiment.')
    manifest['frozen'] = True
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def experiment_folder(root, identity):
    if not identity or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise RagError('Supply the full 64-character experiment manifest ID.')
    folder = root / 'research' / 'experiments' / identity
    if not (folder / 'manifest.json').exists():
        raise RagError('Experiment manifest not found.')
    return folder
