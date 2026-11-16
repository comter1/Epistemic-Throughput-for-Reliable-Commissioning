"""Read immutable source/SED stores; confine all extension writes to its run."""
from pathlib import Path

from cgx.data.store import Store
from cgx.provenance import canonical_hash, read_json
from structured_evidence_v1.storage import OverlayStore


class ExtensionStore(OverlayStore):
    def __init__(self, source_root, sed_run, run_root):
        super().__init__(source_root, run_root)
        self.sed_root=Path(sed_run).resolve()
        if self.root.resolve() in (self.source_root,self.sed_root):
            raise ValueError('Extension output must be separate from immutable inputs')
        self.sed_store=Store(self.sed_root)

    def _get(self,kind,digest):
        if (self.root/'public'/kind/(digest+'.json.gz')).is_file():
            return Store._get(self,kind,digest)
        if (self.sed_root/'public'/kind/(digest+'.json.gz')).is_file():
            return self.sed_store._get(kind,digest)
        return self.source_store._get(kind,digest)

    def _put(self,kind,obj):
        digest=canonical_hash(obj)
        if any((root/'public'/kind/(digest+'.json.gz')).is_file()
               for root in (self.sed_root,self.source_root)):
            return digest
        return Store._put(self,kind,obj)

    def load_provenance(self,history_hash,provenance_hash=None):
        if provenance_hash is not None:
            for root in (self.root,self.sed_root,self.source_root):
                path=root/'public/event_provenance'/history_hash/(provenance_hash+'.json')
                if path.is_file():
                    value=read_json(path)
                    if canonical_hash(value)!=provenance_hash:raise ValueError('Corrupt public provenance')
                    return value
        return super().load_provenance(history_hash,provenance_hash)

    def resolve_blob(self,kind,digest):
        for root in (self.root,self.sed_root,self.source_root):
            path=root/'public'/kind/(digest+'.json.gz')
            if path.is_file():return str(path)
        raise FileNotFoundError((kind,digest))

    def resolve_provenance(self,history_hash,provenance_hash):
        for root in (self.root,self.sed_root,self.source_root):
            path=root/'public/event_provenance'/history_hash/(provenance_hash+'.json')
            if path.is_file():return str(path)
        raise FileNotFoundError((history_hash,provenance_hash))
