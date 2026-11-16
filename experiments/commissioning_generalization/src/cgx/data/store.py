"""Content-addressed, losslessly compressed observations with no truth in history.

Only the orchestrator uses manifest methods. Model calls receive load_history's plain
whitelisted dictionary, never this object or its private root.
"""
from pathlib import Path
import gzip
import json
import os
import tempfile
import math
from functools import lru_cache
import jsonschema
from cgx.provenance import canonical_bytes, canonical_hash, file_hash, atomic_json
from cgx.graphs.manifests import read_jsonl,load_cases

FIELDS = {"graph","observations","remaining","initial_budget","action_keys","nominal_task","requirements"}
RESOURCE_FIELDS = {"executions","bytes","calibrations","decisions"}


def _nominal_contract(task,requirements):
    allowed_task={"waypoints","home","deadlines_s","reference_xyz","hold_seconds","duration_s"}
    if set(task)-allowed_task: raise ValueError("Unexpected nominal task fields")
    def vector(value,size):
        return isinstance(value,list) and len(value)==size and all(type(x) in (int,float) and math.isfinite(x) for x in value)
    if "waypoints" not in task or not isinstance(task["waypoints"],list) or len(task["waypoints"])!=3 or not all(vector(v,7) for v in task["waypoints"]): raise ValueError("Invalid nominal waypoint payload")
    for name,size in (("home",7),("deadlines_s",3)):
        if name in task and not vector(task[name],size): raise ValueError("Invalid nominal task vector")
    allowed_req={"position_tolerance_m","hold_seconds","joint_limit_tolerance_rad","no_prohibited_contacts","command_limits"}
    if set(requirements)-allowed_req: raise ValueError("Unexpected requirements fields")
    if any(type(v) not in (int,float,bool) for v in requirements.values()): raise ValueError("Requirements must be scalar nominal values")


@lru_cache(maxsize=8)
def _schema(name):
    base=Path(__file__).resolve().parents[3]
    return json.loads((base/"schemas"/(name+".schema.json")).read_text())


def check_public_history(history):
    if set(history)!=FIELDS: raise ValueError(f"Public history fields differ: {set(history)^FIELDS}")
    jsonschema.validate(history["graph"],_schema("public_graph"))
    for obs in history["observations"]: jsonschema.validate(obs,_schema("observation"))
    for name in ("remaining","initial_budget"):
        if set(history[name])!=RESOURCE_FIELDS or any(type(v) is not int or v<0 for v in history[name].values()): raise ValueError("Invalid resource vector")
    # These are nominal public contracts, never arbitrary passthrough dictionaries.
    _nominal_contract(history["nominal_task"],history["requirements"])
    return True


def _atomic_gzip(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",compresslevel=1,mtime=0) as z: z.write(payload)
            f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


class Store:
    def __init__(self, root):
        self.root=Path(root)
        self._graphs=None; self._tasks=None; self._configs=None
        self._validated_graphs=set(); self._validated_observations=set()

    def _put(self,kind,obj):
        digest=canonical_hash(obj); path=self.root/"public"/kind/(digest+".json.gz")
        if not path.exists(): _atomic_gzip(path,canonical_bytes(obj))
        return digest

    def _get(self,kind,digest):
        if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest): raise ValueError("Invalid content address")
        path=self.root/"public"/kind/(digest+".json.gz")
        with gzip.open(path,"rt") as f: obj=json.load(f)
        if canonical_hash(obj)!=digest: raise ValueError(f"Corrupt {kind} blob {digest}")
        return obj

    def put_observation(self,observation):
        digest=canonical_hash(observation)
        if digest not in self._validated_observations:
            jsonschema.Draft202012Validator(_schema("observation")).validate(observation)
            self._validated_observations.add(digest)
        return self._put("observations",observation)

    def load_observation(self,digest): return self._get("observations",digest)

    def put_history(self,history):
        if set(history)!=FIELDS: raise ValueError(f"Public history fields differ: {set(history)^FIELDS}")
        # Validate only new blobs; growing histories reuse already checked references.
        graph_hash=canonical_hash(history["graph"])
        if graph_hash not in self._validated_graphs:
            jsonschema.Draft202012Validator(_schema("public_graph")).validate(history["graph"])
            self._validated_graphs.add(graph_hash)
        for name in ("remaining","initial_budget"):
            if set(history[name])!=RESOURCE_FIELDS or any(type(v) is not int or v<0 for v in history[name].values()): raise ValueError("Invalid resources")
        task=history["nominal_task"]; requirements=history["requirements"]
        _nominal_contract(task,requirements)
        packed={"graph_ref":self._put("graphs",history["graph"]),
                "observation_refs":[self.put_observation(o) for o in history["observations"]],
                "task_ref":self._put("tasks",task),"requirements":requirements,
                "remaining":history["remaining"],"initial_budget":history["initial_budget"],"action_keys":history["action_keys"]}
        return self._put("histories",packed)

    def history_references(self,digest): return self._get("histories",digest)

    def load_history(self,digest):
        packed=self._get("histories",digest)
        return {"graph":self._get("graphs",packed["graph_ref"]),
                "observations":[self.load_observation(x) for x in packed["observation_refs"]],
                "nominal_task":self._get("tasks",packed["task_ref"]),"requirements":packed["requirements"],
                "remaining":packed["remaining"],"initial_budget":packed["initial_budget"],"action_keys":packed["action_keys"]}

    def configurations(self,split=None):
        if self._configs is None: self._configs=list(read_jsonl(self.root/"manifests"/"split_manifest.jsonl"))
        return [r for r in self._configs if split is None or r["split"]==split]

    def graph(self,configuration_id):
        if self._graphs is None: self._graphs={r["configuration_id"]:r["graph"] for r in read_jsonl(self.root/"manifests"/"public_graphs.jsonl")}
        return self._graphs[configuration_id]

    def task(self,case_id):
        if self._tasks is None: self._tasks={r["case_id"]:r["task"] for r in read_jsonl(self.root/"manifests"/"tasks.jsonl")}
        return self._tasks[case_id]

    def cases(self,split=None):
        yield from load_cases(self.root,split)

    def transition_path(self,split,case_id): return self.root/"private"/"transitions"/split/(case_id+".jsonl.gz")
    def completion_path(self,split,case_id): return self.root/"jobs"/"collection"/split/(case_id+".json")

    def write_transitions(self,split,case_id,rows):
        path=self.transition_path(split,case_id)
        _atomic_gzip(path,b"".join(canonical_bytes(r)+b"\n" for r in rows))
        return file_hash(path)

    def transitions(self,split="train"):
        directory=self.root/"private"/"transitions"/split
        if not directory.exists(): raise FileNotFoundError(f"No collected split: {split}")
        for path in sorted(directory.glob("*.jsonl.gz")):
            marker=self.completion_path(split,path.name.removesuffix(".jsonl.gz"))
            if not marker.exists(): continue
            status=json.loads(marker.read_text())
            if status.get("state")!="passed": continue
            if file_hash(path)!=status["transition_file_hash"]: raise ValueError("Transition shard hash changed")
            with gzip.open(path,"rt") as f:
                for line in f: yield json.loads(line)


def public_requirements(config):
    return {k:config["system"][k] for k in ("position_tolerance_m","hold_seconds","joint_limit_tolerance_rad")} | {"no_prohibited_contacts":True,"command_limits":True}
