"""Export renamed presentation artifacts without rewriting sealed experiment files.

Run directly with ``python -m cgx.reporting.presentation``. This command only reads
saved results; it never calls aggregation, training, acquisition or model loading.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from cgx.config import EXPERIMENT, load_config
from cgx.provenance import atomic_json, file_sha256, source_hash
from .methods import DISPLAY_NAMES, display_name
from .v2 import figures_v2


def copy_display_csv(source, destination):
    """Add display_name while preserving every existing CSV cell as text."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, [])
        headers = [first]
        if "method" not in first:
            # The original summary.csv has two metric headers and an index header.
            if source.name == "summary.csv":
                headers.extend([next(reader), next(reader)])
            if "method" not in headers[-1]:
                shutil.copy2(source, destination)
                return {"file": source.name, "display_name_added": False}
        method_index = headers[-1].index("method")
        insertion = method_index + 1
        if "display_name" in headers[-1]:
            raise ValueError(f"Source already contains a display_name column: {source}")
        count = 0
        with destination.open("w", newline="") as output:
            writer = csv.writer(output)
            for i, row in enumerate(headers):
                value = "display_name" if i == len(headers) - 1 else ""
                writer.writerow(row[:insertion] + [value] + row[insertion:])
            for row in reader:
                if len(row) != len(headers[-1]):
                    raise ValueError(f"Unexpected CSV row width in {source}")
                writer.writerow(row[:insertion] + [display_name(row[method_index])] + row[insertion:])
                count += 1
    # Compare the exported CSV back to its source, without parsing any numbers.
    with source.open(newline="") as left, destination.open(newline="") as right:
        original, exported = csv.reader(left), csv.reader(right)
        for old, new in zip(original, exported, strict=True):
            if old != new[:insertion] + new[insertion + 1:]:
                raise AssertionError(f"An original CSV cell changed: {source}")
    return {"file": source.name, "display_name_added": True, "data_rows": count,
            "original_cells_identical": True}


def protected_inventory(root):
    """Hash result files, checkpoints and sealed inputs; do not scan raw trajectories."""
    root = Path(root)
    paths = set()
    for directory in ("results", "models", "manifests", "software", "release"):
        paths.update(p for p in (root / directory).rglob("*") if p.is_file())
    for name in ("protocol_lock.json", "environment.json"):
        if (root / name).is_file():
            paths.add(root / name)
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}


def export_presentation(root, output, config=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    # A sibling presentation directory under the data root is allowed. Existing
    # sealed subtrees and the source repository are never valid destinations.
    forbidden = [root, EXPERIMENT.parents[1]]
    forbidden.extend(root / name for name in ("results", "models", "manifests", "software",
                                              "release", "evaluation", "training", "data", "private"))
    for base in forbidden:
        if output == base or (base != root and base in output.parents):
            raise ValueError(f"Presentation output must be separate from protected files: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Presentation output must be a new or empty directory")
    config = load_config() if config is None else config
    if config.get("protocol_id") != "CGX-2.0":
        raise ValueError("This presentation export requires saved CGX-2.0 results")
    before = protected_inventory(root)
    output.mkdir(parents=True, exist_ok=True)
    rendered = figures_v2(root, config, output_dir=output)
    checks = [copy_display_csv(path, output / "tables" / path.name)
              for path in sorted((root / "results").glob("*.csv"))]
    atomic_json(output / "method_display_names.json", DISPLAY_NAMES)
    # Keep a paper-ready description beside the figures without changing the
    # archived source package that generated the scientific results.
    methods_document = EXPERIMENT / "docs" / "METHODS.md"
    if methods_document.is_file():
        methods_text = methods_document.read_text().replace(
            "[`src/cgx/reporting/methods.py`](../src/cgx/reporting/methods.py)",
            "[method_display_names.json](method_display_names.json)").replace(
            "[\u0060models/networks.py\u0060](../src/cgx/models/networks.py)",
            f"[models/networks.py]({EXPERIMENT / 'src/cgx/models/networks.py'})")
        (output / "METHODS.md").write_text(methods_text)
    original_report = root / "results" / "MAIN_EXPERIMENT_V2_RESULTS.md"
    (output / "README.md").write_text(
        "# CGX-2.0 method-name presentation update\n\n"
        "This directory contains display-only derivatives of the completed experiment. "
        "No acquisition, training, checkpoint selection, bootstrap or scientific evaluation was rerun.\n\n"
        "- [Updated main results](MAIN_EXPERIMENT_V2_RESULTS.md)\n"
        "- [Method descriptions](METHODS.md)\n"
        "- [Figure captions](FIGURE_CAPTIONS.md)\n"
        "- [Central display-name mapping](method_display_names.json)\n"
        "- `figures/`: four relabeled main figures, each in PNG and PDF.\n"
        "- `tables/`: copies of saved CSVs; the method key and every original cell are retained. "
        "A display_name column is added wherever a method column exists. "
        "Independent-reader tables retain their separate reader identities.\n"
        "- `presentation_manifest.json`: input/output hashes and preservation checks.\n\n"
        f"The original report at `{original_report}` and the original `results/figures/` "
        "use legacy display names. They, the original CSVs, and the original release archive "
        "are retained unchanged as provenance; use this directory for the corrected presentation.\n\n"
        "The sealed scientific source and protocol hashes still identify the original run. "
        "The current source hash in this presentation manifest identifies only the revised "
        "reporting code; it does not reseal or replace that experiment.\n"
    )
    after = protected_inventory(root)
    if before != after:
        raise AssertionError("A protected experiment artifact changed during presentation export")
    outputs = {str(p.relative_to(output)): file_sha256(p)
               for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {
        "state": "passed", "scope": "display_names_and_descriptions_only",
        "input_root": str(root), "output_root": str(output),
        "scientific_experiment_rerun": False, "protected_artifacts_unchanged": True,
        "protected_file_count": len(before), "protected_input_sha256": before,
        "reporting_source_hash": source_hash(EXPERIMENT),
        "method_display_names": DISPLAY_NAMES, "csv_checks": checks,
        "output_sha256": outputs,
    }
    atomic_json(output / "presentation_manifest.json", manifest)
    return {**rendered, "manifest": str(output / "presentation_manifest.json"),
            "csv_tables": len(checks), "protected_file_count": len(before)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_presentation(args.data_root, args.out, load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
