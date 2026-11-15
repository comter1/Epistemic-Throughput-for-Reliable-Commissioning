"""Pre-training protocol and software seal."""
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET
from cgx.config import EXPERIMENT
from cgx.provenance import atomic_json, read_json, file_sha256, hash_json, source_hash, now

def lock_protocol(root,config):
    root=Path(root)
    if (root/"protocol_lock.json").exists():
        from cgx.provenance import require_protocol_lock
        return require_protocol_lock(root,config)
    required=["environment.json","reports/qualification.json","reports/manifest_validation.json",
              "reports/legacy_integrity.json","reports/source_healthy_catalog.json",
              "reports/technical_smoke.json","reports/resource_forecast.json"]
    for name in required:
        report=read_json(root/name)
        if report.get("state")!="passed": raise ValueError(f"Unpassed prerequisite: {name}")
    if read_json(root/"reports/technical_smoke.json").get("source_hash")!=source_hash(EXPERIMENT):
        raise ValueError("Technical pipeline/runtime evidence must match final source before sealing")
    tree=ET.parse(root/"reports/pytest.xml")
    tests=list(tree.iter("testcase"))
    if not tests or any(t.find("failure") is not None or t.find("error") is not None or t.find("skipped") is not None for t in tests):
        raise ValueError("Engineering test suite must actually pass before sealing")
    software=root/"software"; software.mkdir(parents=True,exist_ok=True)
    shutil.copy2(EXPERIMENT/"requirements.lock.txt",software/"requirements.lock.txt")
    # The active, versioned profile may differ from the untouched full protocol.
    atomic_json(software/"protocol.json",config)
    source_files={str(p.relative_to(EXPERIMENT)):file_sha256(p) for directory in ["src","schemas","configs","contracts","tests","docs","scripts","tools","examples"]
                  for p in sorted((EXPERIMENT/directory).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    for filename in ("pyproject.toml","AGENTS.md","requirements.lock.txt"):
        source_files[filename]=file_sha256(EXPERIMENT/filename)
    atomic_json(software/"source_files.json",source_files)
    with tarfile.open(software/"source.tar.gz","w:gz") as archive:
        for relative in source_files: archive.add(EXPERIMENT/relative,arcname=relative)
    files=read_json(root/"manifests/hashes.json")
    sealed_private_test_files={key:value for key,value in files.items()
                               if key=="private/private_cases.jsonl" or key.startswith("private/cases/test_")}
    files={key:value for key,value in files.items() if key not in sealed_private_test_files}
    for name in required+["reports/pytest.xml","software/requirements.lock.txt","software/protocol.json","software/source_files.json","software/source.tar.gz"]:
        files[name]=file_sha256(root/name)
    environment=read_json(root/"environment.json")
    files.update(environment["asset_hashes"])
    record={"state":"passed","locked_at":now(),"protocol_id":config["protocol_id"],"config_hash":hash_json(config),
            "source_hash":source_hash(EXPERIMENT),"files":files,"sealed_private_test_files":sealed_private_test_files,
            "asset_git_sha":environment["asset_git_sha"],
            "test_outcomes_accessed":False,"training_started":False,"technical_smoke":False,
            "validation_policy_selection":{
                "cases":config['splits']['validation']['configurations']*config['splits']['validation']['cases_per_configuration'],
                "byte_fraction":config['evaluation']['primary_byte_fraction'],
                "execution_cap":config['evaluation']['main_execution_cap'],
                "ranking":"final NLL, then actual bytes, then earlier checkpoint",
                "early_stopping":config['training'].get('early_stopping')},
            "retention":"All public acquired JSON losslessly compressed; binary source wire deterministic2%; all evaluation evidence retained"}
    atomic_json(root/"protocol_lock.json",record);atomic_json(EXPERIMENT/"reports/protocol_lock.json",record)
    return record
