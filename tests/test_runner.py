from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scCompass
from scCompass.exceptions import SolverUnavailableError
from scCompass._compat_core import translate_solver_error
from scCompass.runner import run_compass


def _source():
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            [
                [1.0, 10.0, 5.0],
                [2.0, 20.0, 6.0],
                [3.0, 30.0, 7.0],
            ]
        ),
        obs=pd.DataFrame({"celltype": ["T", "B", "T"]}, index=["c1", "c2", "c3"]),
        layers={"counts": sparse.csr_matrix(
            [
                [1.0, 10.0, 5.0],
                [2.0, 20.0, 6.0],
                [3.0, 30.0, 7.0],
            ]
        )},
    )
    adata.var_names = ["G1", "G2", "G3"]
    return adata


def _model():
    reaction = SimpleNamespace(list_genes=lambda: {"G1"})
    return SimpleNamespace(
        name="TEST_MODEL",
        reactions={"R1": reaction},
        media="test-media",
    )


def test_runner_maps_adata_to_core_and_builds_result():
    calls = {}

    def core_runner(expression, model, media, species, args):
        calls["shape"] = expression.shape
        calls["cells"] = expression.columns.tolist()
        calls["lambda"] = args["lambda"]
        scores = pd.DataFrame(
            [[0.1, 0.2, 0.3]],
            index=["R1"],
            columns=["c3", "c1", "c2"],
        )
        uptake = pd.DataFrame(
            [[1.0, 2.0, 3.0]], index=["M1"], columns=["c3", "c1", "c2"]
        )
        return {"reactions": scores, "uptake": uptake}

    source = _source()
    source.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = run_compass(
        source,
        model="TEST_MODEL",
        expression_layer=None,
        lambda_=0.25,
        model_loader=lambda name, species: _model(),
        core_runner=core_runner,
    )

    assert calls == {"shape": (1, 3), "cells": ["c1", "c2", "c3"], "lambda": 0.25}
    assert result.obs_names.tolist() == ["c1", "c2", "c3"]
    assert result.uns["scCompass_reaction_columns"].tolist() == ["R1"]
    assert result.var_names.tolist() == ["G1", "G2", "G3"]
    np.testing.assert_allclose(
        result.obsm["scCompass_reactions"][:, 0], [0.2, 0.3, 0.1]
    )
    np.testing.assert_allclose(result.obsm["scCompass_uptake"][:, 0], [2.0, 3.0, 1.0])
    assert result.uns["scCompass"]["model"] == "TEST_MODEL"
    assert result.uns["scCompass"]["expression_layer"] == "X"


def test_injected_runner_is_used_by_public_api():
    matrix = sparse.csr_matrix(
        [
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 6.0],
            [3.0, 30.0, 7.0],
            [4.0, 40.0, 8.0],
        ]
    )
    source = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["c1", "c2", "c3", "c4"]),
    )

    def core_runner(expression, model, media, species, args):
        return {
            "reactions": pd.DataFrame(
                [[1.0, 2.0, 3.0, 4.0]],
                index=["R1"],
                columns=["c1", "c2", "c3", "c4"],
            )
        }

    result = scCompass.compass(
        source,
        expression_layer="counts",
        model_loader=lambda name, species: _model(),
        core_runner=core_runner,
    )

    assert result.shape == (4, 3)


def test_default_core_reports_missing_cplex_clearly():
    try:
        import cplex  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CPLEX is installed")

    with pytest.raises(SolverUnavailableError, match="CPLEX"):
        scCompass.compass(
            _source(),
            expression_layer="counts",
            model_loader=lambda name, species: _model(),
        )


def test_cplex_community_size_limit_is_explained():
    class SolverError(Exception):
        pass

    error = SolverError(
        "CPLEX Error  1016: Community Edition. Problem size limits exceeded."
    )

    with pytest.raises(SolverUnavailableError, match="Community Edition"):
        translate_solver_error(error)
