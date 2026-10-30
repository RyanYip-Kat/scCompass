import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd

from scCompass.result import build_result


SCRIPT = "scripts/pipeline_vis.py"


def _write_result(path):
    source = ad.AnnData(
        X=np.zeros((6, 2)),
        obs=pd.DataFrame(
            {"celltype": ["A", "A", "B", "B", "C", "C"]},
            index=[f"c{i}" for i in range(6)],
        ),
    )
    source.obsm["X_umap"] = np.column_stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)]
    )
    scores = pd.DataFrame(
        {
            "R1": [1, 1.1, 2, 2.1, 3, 3.1],
            "R2": [3, 3.1, 2, 2.1, 1, 1.1],
        },
        index=source.obs_names,
    )
    metadata = pd.DataFrame(
        {
            "subsystem": ["Glycolysis", "TCA"],
            "reactants": ["M1", "M2"],
            "products": ["M2", "M3"],
            "is_exchange": [False, False],
        },
        index=["R1", "R2"],
    )
    uptake = pd.DataFrame(
        {"M1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        index=source.obs_names,
    )
    secretion = pd.DataFrame(
        {"M2": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        index=source.obs_names,
    )
    result = build_result(
        source,
        scores,
        reaction_metadata=metadata,
        uptake=uptake,
        secretion=secretion,
    )
    result.write_h5ad(path)


def test_pipeline_vis_generates_report_from_saved_anndata(tmp_path):
    input_path = tmp_path / "result.h5ad"
    output_dir = tmp_path / "report"
    _write_result(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--groupby",
            "celltype",
            "--top-n",
            "2",
            "--format",
            "png",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(
        (output_dir / "visual_report_manifest.json").read_text()
    )
    assert manifest["groupby"] == "celltype"
    assert manifest["outputs"]["heatmap"].endswith("reaction_heatmap.png")
    assert manifest["outputs"]["umap"].endswith("reaction_umap.png")
    assert manifest["outputs"]["metabolite_volcano"].endswith(
        "metabolite_volcano.png"
    )
    assert manifest["outputs"]["metabolite_pathway_strip"].endswith(
        "metabolite_pathway_strip.png"
    )
    assert (output_dir / "reaction_differential.tsv").exists()


def test_pipeline_vis_help_lists_choices():
    completed = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--groupby" in completed.stdout
    assert "--include-exchange" in completed.stdout
