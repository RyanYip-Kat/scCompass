import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scCompass
from scCompass.core.models.MetabolicModel import (
    Association,
    Gene,
    MetabolicModel,
    Reaction,
    Species,
)


def _gene(name):
    gene = Gene()
    gene.id = name
    gene.name = name
    return gene


def _gene_association(name):
    association = Association()
    association.type = "gene"
    association.gene = _gene(name)
    return association


def _tiny_model():
    m1, m2, m3 = Species(), Species(), Species()
    m1.id, m2.id, m3.id = "M1", "M2", "M3"

    e1 = Reaction()
    e1.id, e1.lower_bound, e1.upper_bound = "E1", 0.0, 10.0
    e1.products["M1"] = 1.0

    r1 = Reaction()
    r1.id, r1.lower_bound, r1.upper_bound = "R1", 0.0, 10.0
    r1.reactants["M1"] = 1.0
    r1.products["M2"] = 1.0
    r1.gene_associations = _gene_association("G1")

    r2 = Reaction()
    r2.id, r2.lower_bound, r2.upper_bound = "R2", -10.0, 10.0
    r2.reactants["M2"] = 1.0
    r2.products["M3"] = 1.0
    r2.gene_associations = _gene_association("G2")

    e3 = Reaction()
    e3.id, e3.lower_bound, e3.upper_bound = "E3", 0.0, 10.0
    e3.reactants["M3"] = 1.0

    model = MetabolicModel("TinyCompassModel")
    model.reactions = {r.id: r for r in [e1, r1, r2, e3]}
    model.species = {m.id: m for m in [m1, m2, m3]}
    model.limitExchangeReactions(1.0)
    model.make_unidirectional()
    return model


def test_real_compass_core_runs_on_adata(tmp_path, monkeypatch):
    from scCompass.core.compass import cache

    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(tmp_path / "cache"))
    cache._cache.clear()
    cache._new_cache.clear()

    matrix = sparse.csr_matrix([[2.0, 1.0], [1.0, 2.0]])
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["cell1", "cell2"]),
    )
    adata.var_names = ["G1", "G2"]
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    model = _tiny_model()

    result = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=False,
        model_loader=lambda name, species: model,
    )

    assert result.shape[0] == 2
    assert set(result.uns["scCompass_reaction_columns"]).issuperset(
        {"R1_pos", "R2_pos", "R2_neg"}
    )
    assert np.isfinite(result.obsm["scCompass_reactions"]).all()
    assert result.uns["scCompass"]["core_args"]["penalty_backend"] == "vectorized"
    np.testing.assert_array_equal(result.obsm["X_umap"], adata.obsm["X_umap"])


def test_real_core_returns_uptake_and_secretion_in_result_obsm(tmp_path, monkeypatch):
    from scCompass.core.compass import cache

    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(tmp_path / "cache"))
    cache._cache.clear()
    cache._new_cache.clear()

    matrix = sparse.csr_matrix([[2.0, 1.0], [1.0, 2.0]])
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["cell1", "cell2"]),
    )
    adata.var_names = ["G1", "G2"]
    model = _tiny_model()

    result = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=True,
        model_loader=lambda name, species: model,
    )

    assert result.obsm["scCompass_uptake"].shape[0] == 2
    assert result.obsm["scCompass_secretion"].shape[0] == 2
    assert result.uns["scCompass"]["outputs"]["uptake"] == "scCompass_uptake"


def test_checkpoint_resume_skips_completed_cells(tmp_path, monkeypatch):
    from scCompass.core.compass import cache

    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(tmp_path / "cache"))
    cache._cache.clear()
    cache._new_cache.clear()

    matrix = sparse.csr_matrix([[2.0, 1.0], [1.0, 2.0]])
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["cell1", "cell2"]),
    )
    adata.var_names = ["G1", "G2"]
    model = _tiny_model()
    checkpoint_dir = tmp_path / "checkpoint"

    first = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=True,
        model_loader=lambda name, species: model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_batch_size=1,
    )
    assert first.uns["scCompass"]["checkpoint"]["completed_before"] == 0
    assert first.uns["scCompass"]["checkpoint"]["computed"] == 2
    assert len(list(checkpoint_dir.glob("batch-*.npz"))) == 2
    assert not list(checkpoint_dir.rglob("*.tsv"))

    import scCompass.core.compass.algorithm as algorithm

    def fail_if_solved(*args, **kwargs):
        raise AssertionError("completed cells must not be solved again")

    monkeypatch.setattr(algorithm, "compass_reactions", fail_if_solved)
    second = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=True,
        model_loader=lambda name, species: model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_batch_size=1,
    )

    assert second.uns["scCompass"]["checkpoint"]["completed_before"] == 2
    assert second.uns["scCompass"]["checkpoint"]["computed"] == 0
    np.testing.assert_allclose(
        first.obsm["scCompass_reactions"], second.obsm["scCompass_reactions"]
    )
    np.testing.assert_allclose(
        first.obsm["scCompass_uptake"], second.obsm["scCompass_uptake"]
    )


def test_checkpoint_resumes_after_failure_mid_run(tmp_path, monkeypatch):
    from scCompass.core.compass import cache
    import scCompass.core.compass.algorithm as algorithm

    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(tmp_path / "cache"))
    cache._cache.clear()
    cache._new_cache.clear()

    matrix = sparse.csr_matrix([[2.0, 1.0], [1.0, 2.0]])
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["cell1", "cell2"]),
    )
    adata.var_names = ["G1", "G2"]
    model = _tiny_model()
    checkpoint_dir = tmp_path / "checkpoint"
    original = algorithm.compass_reactions
    calls = {"count": 0}

    def fail_on_second_cell(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithm, "compass_reactions", fail_on_second_cell)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        scCompass.compass(
            adata,
            model=model.name,
            expression_layer="counts",
            lambda_=0.0,
            calc_metabolites=True,
            model_loader=lambda name, species: model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_batch_size=1,
        )

    monkeypatch.setattr(algorithm, "compass_reactions", original)
    resumed = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=True,
        model_loader=lambda name, species: model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_batch_size=1,
    )

    assert resumed.shape[0] == 2
    assert resumed.uns["scCompass"]["checkpoint"]["completed_before"] == 1
    assert resumed.uns["scCompass"]["checkpoint"]["computed"] == 1
