"""Independent CPU rollout workers with frozen weights and real per-call deadlines.

Each worker runs the existing evaluator in its main thread. No policy receives a
private case or worker context. GPU training remains one independent job per GPU.
CPU inference is explicitly identified in every returned timing record.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import multiprocessing as mp
import os
import torch

from cgx.baselines import RuntimePolicy
from cgx.models import DiagnosticReader, PolicyModel
from cgx.training.objectives import frozen
from .rollout import rollout

_CONTEXT = None


def evaluation_workers():
    from cgx.config import allocated_cpu_count
    return min(max(1, int(os.environ.get("CGX_EVAL_WORKERS", "1"))), allocated_cpu_count())


def model_spec(model):
    if model is None:
        return None
    if isinstance(model, DiagnosticReader):
        init = {"architecture": model.architecture, "width": model.width, "floor": model.floor}
        kind = "reader"
    else:
        init = {"method": model.method, "width": model.width}
        kind = "policy"
    return {"kind": kind, "init": init,
            "weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def _construct(spec):
    if spec is None:
        return None
    model = (DiagnosticReader if spec["kind"] == "reader" else PolicyModel)(**spec["init"])
    model.load_state_dict(spec["weights"])
    return frozen(model)


def _initialize(reader_spec, policy_spec, method, beta, config, asset_path, store_root):
    global _CONTEXT
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from cgx.data.store import Store
    reader = _construct(reader_spec)
    policy = RuntimePolicy(method, _construct(policy_spec), beta)
    _CONTEXT = (reader, policy, config, asset_path, Store(store_root) if store_root else None)


def _run(job):
    reader, policy, config, asset_path, store = _CONTEXT
    graph, case, task, fraction, cap, implementation, unrestricted = job[:7]
    row, traces, _, system = rollout(graph, case, task, config, reader, policy, fraction, cap,
                               implementation, asset_path, store, unrestricted=unrestricted)
    if len(job)>7 and job[7]=='validation_repair':
        import numpy as np
        from .repairs import verification
        confidence=float(np.exp2(max(row['prediction']['log_probabilities_base2'])))
        outcome=verification(system,case,config,row['predicted_target_key'],confidence,min(config['evaluation']['repair_threshold_grid']),row['abstained'],audit=True)
        row['_verification']={'confidence':confidence,'abstained':row['abstained'],'public_pass':outcome['public_pass'],'audit_all_pass':outcome['audit_all_pass']}
    row.update(inference_device="cpu", rollout_execution="independent_process",
               cpu_threads_per_worker=1)
    return row, traces


@contextmanager
def rollout_pool(reader, policy_model, method, config, asset_path, *, beta=.5,
                 store_root=None, workers=None):
    # spawn never inherits the trainer's live CUDA context or mutable optimizer.
    with ProcessPoolExecutor(max_workers=workers or evaluation_workers(),
            mp_context=mp.get_context("spawn"), initializer=_initialize,
            initargs=(model_spec(reader), model_spec(policy_model), method, beta,
                      config, str(asset_path), str(store_root) if store_root else None)) as pool:
        yield pool


def map_rollouts(pool, jobs):
    # Bounded chunks avoid millions of queued Future objects. Ordered results
    # preserve the configuration/case aggregation order of serial validation.
    import itertools
    iterator = iter(jobs)
    while batch := list(itertools.islice(iterator, 512)):
        yield from pool.map(_run, batch, chunksize=4)
