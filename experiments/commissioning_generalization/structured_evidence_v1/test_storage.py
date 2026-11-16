from pathlib import Path
import pytest
from cgx.data.store import Store
from cgx.provenance import canonical_hash
from structured_evidence_v1.storage import OverlayStore


def test_readthrough_and_writes_are_separate(tmp_path):
    source, run = tmp_path / 'old', tmp_path / 'new'
    old = Store(source)
    first = {'public': 'existing observation'}
    key = old._put('observations', first)
    overlay = OverlayStore(source, run)
    assert overlay._get('observations', key) == first
    assert overlay._put('observations', first) == key
    assert not (run / 'public/observations' / (key + '.json.gz')).exists()
    second = {'public': 'new observation'}
    second_key = overlay._put('observations', second)
    assert overlay._get('observations', second_key) == second
    assert not (source / 'public/observations' / (second_key + '.json.gz')).exists()


def test_provenance_distinguishes_same_values_different_events(tmp_path):
    overlay = OverlayStore(tmp_path / 'old', tmp_path / 'new')
    history_hash = canonical_hash({'same': 'raw observation values'})
    duplicate = {'observation_ids': ['acquired-A', 'acquired-A']}
    independent = {'observation_ids': ['acquired-A', 'acquired-B']}
    d = overlay.save_provenance(history_hash, duplicate)
    n = overlay.save_provenance(history_hash, independent)
    assert d != n
    assert overlay.load_provenance(history_hash, d) == duplicate
    assert overlay.load_provenance(history_hash, n) == independent
    with pytest.raises(ValueError, match='provenance versions'):
        overlay.load_provenance(history_hash)


def test_old_data_cannot_be_writable_output(tmp_path):
    source = tmp_path / 'old'
    with pytest.raises(ValueError, match='own output'):
        OverlayStore(source, source)
    run = tmp_path / 'new'; run.mkdir()
    (run / 'public').symlink_to(source / 'public')
    with pytest.raises(ValueError, match='must not link'):
        OverlayStore(source, run)
