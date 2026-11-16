"""Read-through original evidence with writes confined to the new experiment.

Provenance is a separate public identity envelope. It never changes the original
wire schema or gives fault labels, simulator time, or private clock offsets to a
reader. A raw content hash alone cannot distinguish duplicate storage from two
different events with identical values, so provenance has its own content key.
"""
from pathlib import Path
import fcntl
from cgx.data.store import Store
from cgx.provenance import canonical_hash, atomic_json, read_json


class OverlayStore(Store):
    def __init__(self, source_root, run_root):
        self.source_root = Path(source_root).resolve()
        self.run_root = Path(run_root).resolve()
        if self.run_root == self.source_root:
            raise ValueError("New experiment requires its own output directory")
        public = self.run_root / 'public'
        if public.is_symlink():
            raise ValueError("Writable public directory must not link to original data")
        super().__init__(self.run_root)
        self.source_store = Store(self.source_root)

    def _get(self, kind, digest):
        path = self.root / 'public' / kind / (digest + '.json.gz')
        return super()._get(kind, digest) if path.exists() else self.source_store._get(kind, digest)

    def _put(self, kind, obj):
        digest = canonical_hash(obj)
        if (self.source_root / 'public' / kind / (digest + '.json.gz')).exists():
            return digest
        return super()._put(kind, obj)

    def configurations(self, split=None):
        return self.source_store.configurations(split)

    def graph(self, configuration_id):
        return self.source_store.graph(configuration_id)

    def task(self, case_id):
        return self.source_store.task(case_id)

    def cases(self, split=None):
        return self.source_store.cases(split)

    def save_provenance(self, history_hash, provenance):
        """Return a versioned key; never silently merge conflicting identities."""
        if not isinstance(provenance, dict) or 'observation_ids' not in provenance:
            raise ValueError("Public acquisition identity envelope required")
        if not all(isinstance(v, str) and v for v in provenance['observation_ids']):
            raise ValueError("Acquisition identities must be nonempty opaque strings")
        if len(history_hash) != 64 or any(c not in '0123456789abcdef' for c in history_hash):
            raise ValueError("Invalid history content address")
        digest = canonical_hash(provenance)
        directory = self.root / 'public' / 'event_provenance' / history_hash
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (digest + '.json')
        with (directory / '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if path.exists():
                if read_json(path) != provenance:
                    raise ValueError("Provenance content address collision")
            else:
                atomic_json(path, provenance)
        return digest

    def load_provenance(self, history_hash, provenance_hash=None):
        directory = self.root / 'public' / 'event_provenance' / history_hash
        if provenance_hash is None:
            paths = sorted(directory.glob('*.json'))
            if len(paths) != 1:
                raise ValueError(f"History has {len(paths)} provenance versions; specify its provenance hash")
            path = paths[0]
        else:
            path = directory / (provenance_hash + '.json')
        value = read_json(path)
        if canonical_hash(value) != path.stem:
            raise ValueError("Corrupt provenance envelope")
        return value

    def put_history_with_provenance(self, history, provenance):
        if len(history['observations']) != len(provenance['observation_ids']):
            raise ValueError("Every observation needs an acquisition identity")
        history_hash = self.put_history(history)
        provenance_hash = self.save_provenance(history_hash, provenance)
        return {'history_hash': history_hash, 'provenance_hash': provenance_hash}
