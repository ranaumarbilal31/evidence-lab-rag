import json
from dataclasses import replace

import pytest

from rag.api import fingerprint
from rag.config import Settings, Limits
from rag.dataset import generate_cases
from rag.models import RagError
from rag.research import experiment, freeze, require_reviews, research_lock, validate_cases


def test_assistant_review_never_satisfies_human_gate_and_stale_review_rejected(tmp_path):
    case = generate_cases()[0]
    row = dict(case_id=case['id'], case_fingerprint=fingerprint(case), reviewed=True,
               verdict='accepted', rationale='Question and policy inspected.')
    (tmp_path / 'assistant-label-reviews.jsonl').write_text(json.dumps(row))
    assert require_reviews(tmp_path, [case], 'assistant')[0] == [case]
    with pytest.raises(RagError):
        require_reviews(tmp_path, [case], 'human')
    case['question'] += ' Changed'
    with pytest.raises(RagError):
        require_reviews(tmp_path, [case], 'assistant')


def test_manifest_isolates_changes_and_gates_held_out(tmp_path):
    (tmp_path / 'rag').mkdir()
    code = tmp_path / 'rag' / 'pipeline.py'
    code.write_text('version = 1')
    settings = Settings('private-secret', True, Limits(5, 60000, 20), Limits(10, 60000, 100))
    cases = generate_cases()
    folder, identity = experiment(tmp_path, cases, settings, 'assistant')
    assert 'private-secret' not in (folder / 'manifest.json').read_text()
    assert (folder / 'source/rag/pipeline.py').read_text() == 'version = 1'
    with pytest.raises(RagError, match='frozen'):
        experiment(tmp_path, cases, settings, 'assistant', identity, True)
    with pytest.raises(RagError, match='unfinished'):
        freeze(folder)
    code.write_text('version = 2')
    with pytest.raises(RagError, match='changed'):
        experiment(tmp_path, cases, settings, 'assistant', identity)
    assert experiment(tmp_path, cases, settings, 'assistant')[1] != identity


def test_os_lock_excludes_concurrent_cli_and_releases(tmp_path):
    with research_lock(tmp_path):
        with pytest.raises(RagError, match='Another'):
            with research_lock(tmp_path):
                pass
    with research_lock(tmp_path):
        pass


def test_disputed_labels_excluded_with_denominator(tmp_path):
    cases = generate_cases()[:2]
    cases[0]['human_label_reviewed'] = True
    selected, excluded = require_reviews(tmp_path, cases, 'human')
    assert selected == [cases[0]] and excluded == [cases[1]['id']]
    validate_cases(generate_cases())


def test_freeze_requires_bound_complete_scores(tmp_path):
    import csv
    folder = tmp_path
    (folder / 'manifest.json').write_text(json.dumps({'id': 'experiment', 'review_source': 'assistant'}))
    (folder / 'cases.jsonl').write_text(json.dumps({'id': 'case', 'split': 'development'}))
    (folder / 'runs').mkdir()
    scores = []
    for mode in ('baseline', 'protected', 'safety_wall'):
        name = mode + '.json'
        (folder / 'runs' / name).write_text(json.dumps(dict(case_id='case', mode=mode,
            category='malicious', split='development', manifest_id='experiment',
            review_source='assistant', result={'status': 'answer'})))
        scores.append(dict(run_file=name, reviewed='1', answer_correct='1', citations_supported='1',
            observed_conflict='0', observed_abstention='0', benign_false_positive='0', attack_succeeded=''))
    def write_scores():
        with (folder / 'assistant-scores.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=scores[0].keys())
            writer.writeheader()
            writer.writerows(scores)
    write_scores()
    with pytest.raises(RagError, match='binary'):
        freeze(folder)
    for score in scores:
        score['attack_succeeded'] = '0'
    scores[0]['run_file'] = 'unrelated.json'
    write_scores()
    with pytest.raises(RagError, match='every development'):
        freeze(folder)
    scores[0]['run_file'] = 'baseline.json'
    write_scores()
    freeze(folder)
    assert json.loads((folder / 'manifest.json').read_text())['frozen'] is True
