import gzip
import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scCompass.pipeline import run_pipeline
from scCompass.result import build_result


def _write_input(path):
    matrix = sparse.csr_matrix(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
        ]
    )
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(
            {"celltype": ["T", "B", "T", "B"]},
            index=["c1", "c2", "c3", "c4"],
        ),
    )
    adata.var_names = ["G1", "G2"]
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    adata.write_h5ad(path)


def test_pipeline_writes_complete_artifact_set(tmp_path):
    input_path = tmp_path / "input.h5ad"
    output_dir = tmp_path / "pipeline"
    _write_input(input_path)
    calls = []

    def compass_runner(adata, **options):
        calls.append(options)
        scores = pd.DataFrame(
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
            index=adata.obs_names,
            columns=["R1", "R2"],
        )
        uptake = pd.DataFrame(
            [[1.0], [2.0], [3.0], [4.0]],
            index=adata.obs_names,
            columns=["M1"],
        )
        return build_result(
            adata,
            scores,
            uptake=uptake,
            run_metadata={"solver_backend": "test"},
        )

    artifacts = run_pipeline(
        input_path,
        output_dir,
        compass_runner=compass_runner,
        base_compass_options={"lambda_": 0.0},
        plot_top_reactions=1,
        export_full_table=True,
        overwrite=True,
    )

    assert calls[0]["expression_layer"] == "counts"
    assert calls[0]["checkpoint_dir"] == output_dir / "checkpoint"
    assert artifacts["result"].exists()
    assert artifacts["score_result"].exists()
    assert artifacts["manifest"].exists()
    assert artifacts["reaction_summary"].exists()
    assert artifacts["cell_summary"].exists()
    assert artifacts["plot"].exists()
    assert artifacts["full_table"].exists()

    manifest = json.loads(artifacts["manifest"].read_text())
    assert manifest["status"] == "success"
    assert manifest["result"]["n_obs"] == 4
    assert manifest["result"]["n_reactions"] == 2

    reaction_summary = pd.read_csv(
        artifacts["reaction_summary"], sep="\t", index_col=0
    )
    assert reaction_summary.index.tolist() == ["R1", "R2"]
    assert "mean" in reaction_summary

    with gzip.open(artifacts["full_table"], "rt") as handle:
        assert handle.readline().startswith("\tR1\tR2")


def test_pipeline_skips_completed_output_without_overwrite(tmp_path):
    input_path = tmp_path / "input.h5ad"
    output_dir = tmp_path / "pipeline"
    _write_input(input_path)
    calls = {"count": 0}

    def compass_runner(adata, **options):
        calls["count"] += 1
        scores = pd.DataFrame(
            np.ones((adata.n_obs, 1)),
            index=adata.obs_names,
            columns=["R1"],
        )
        return build_result(adata, scores)

    first = run_pipeline(
        input_path,
        output_dir,
        compass_runner=compass_runner,
        plot_top_reactions=1,
        overwrite=True,
    )
    second = run_pipeline(
        input_path,
        output_dir,
        compass_runner=compass_runner,
        plot_top_reactions=1,
    )

    assert calls["count"] == 1
    assert second["manifest"] == first["manifest"]


def test_pipeline_rejects_completed_output_with_different_config(tmp_path):
    input_path = tmp_path / "input.h5ad"
    output_dir = tmp_path / "pipeline"
    _write_input(input_path)

    def compass_runner(adata, **options):
        scores = pd.DataFrame(
            np.ones((adata.n_obs, 1)),
            index=adata.obs_names,
            columns=["R1"],
        )
        return build_result(adata, scores)

    run_pipeline(
        input_path,
        output_dir,
        compass_runner=compass_runner,
        base_compass_options={"lambda_": 0.0},
        overwrite=True,
    )

    with pytest.raises(ValueError, match="different pipeline configuration"):
        run_pipeline(
            input_path,
            output_dir,
            compass_runner=compass_runner,
            base_compass_options={"lambda_": 0.5},
        )


def test_pipeline_cli_help_lists_full_dataset_options():
    completed = subprocess.run(
        [sys.executable, "-m", "scCompass.pipeline_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--sample-size" in completed.stdout
    assert "--solver-backend" in completed.stdout


def test_pipeline_handles_all_pbmc_cells(tmp_path):
    input_path = Path("testdata/pbmc5k_querydata.h5ad")
    output_dir = tmp_path / "full_pbmc_pipeline"

    def compass_runner(adata, **options):
        scores = pd.DataFrame(
            np.zeros((adata.n_obs, 2), dtype=float),
            index=adata.obs_names,
            columns=["R1", "R2"],
        )
        return build_result(adata, scores)

    artifacts = run_pipeline(
        input_path,
        output_dir,
        compass_runner=compass_runner,
        plot_top_reactions=1,
        overwrite=True,
    )
    manifest = json.loads(artifacts["manifest"].read_text())

    assert manifest["result"]["n_obs"] == 58677
    assert manifest["result"]["n_reactions"] == 2
    assert artifacts["result"].exists()
    assert artifacts["plot"].exists()
