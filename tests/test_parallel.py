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


pytest.importorskip("cplex")


def _tiny_model():
    m1, m2, m3 = Species(), Species(), Species()
    m1.id, m2.id, m3.id = "M1", "M2", "M3"

    inbound = Reaction()
    inbound.id, inbound.lower_bound, inbound.upper_bound = "EX_IN", 0.0, 10.0
    inbound.products["M1"] = 1.0

    linked = []
    for index, gene_name in enumerate(["G1", "G2"]):
        reaction = Reaction()
        reaction.id = f"R{index + 1}"
        reaction.lower_bound, reaction.upper_bound = 0.0, 10.0
        reaction.reactants["M1" if index == 0 else "M2"] = 1.0
        reaction.products["M2" if index == 0 else "M3"] = 1.0
        gene = Gene()
        gene.id = gene.name = gene_name
        association = Association()
        association.type, association.gene = "gene", gene
        reaction.gene_associations = association
        linked.append(reaction)

    outbound = Reaction()
    outbound.id, outbound.lower_bound, outbound.upper_bound = "EX_OUT", 0.0, 10.0
    outbound.reactants["M3"] = 1.0

    model = MetabolicModel("ParallelTestModel")
    model.reactions = {
        reaction.id: reaction
        for reaction in [inbound, *linked, outbound]
    }
    model.species = {metabolite.id: metabolite for metabolite in [m1, m2, m3]}
    model.limitExchangeReactions(1.0)
    model.make_unidirectional()
    return model


def _adata():
    matrix = sparse.csr_matrix(
        [
            [4.0, 1.0],
            [3.0, 2.0],
            [2.0, 3.0],
            [1.0, 4.0],
        ]
    )
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["c1", "c2", "c3", "c4"]),
    )
    adata.var_names = ["G1", "G2"]
    adata.obsm["X_umap"] = np.arange(8, dtype=float).reshape(4, 2)
    return adata


def _run(tmp_path, workers, checkpoint_name=None):
    from scCompass.core.compass import cache

    cache.PREPROCESS_CACHE_DIR = str(tmp_path / "cache")
    cache._cache.clear()
    cache._new_cache.clear()
    model = _tiny_model()
    checkpoint_dir = (
        None if checkpoint_name is None else tmp_path / checkpoint_name
    )
    return scCompass.compass(
        _adata(),
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=True,
        model_loader=lambda name, species: model,
        num_processes=workers,
        num_threads=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_batch_size=1,
    )


def test_process_workers_match_serial_scores(tmp_path):
    serial = _run(tmp_path, workers=1, checkpoint_name="serial_checkpoint")
    parallel = _run(tmp_path, workers=2, checkpoint_name="parallel_checkpoint")

    np.testing.assert_allclose(
        serial.obsm["scCompass_reactions"],
        parallel.obsm["scCompass_reactions"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        serial.obsm["scCompass_uptake"],
        parallel.obsm["scCompass_uptake"],
        rtol=0,
        atol=1e-12,
    )
    assert parallel.obs_names.tolist() == ["c1", "c2", "c3", "c4"]
    assert parallel.uns["scCompass"]["parallel"]["workers"] == 2
    assert parallel.uns["scCompass"]["parallel"]["tasks"] == 4


def test_parallel_workers_share_atomic_checkpoint_batches(tmp_path):
    result = _run(tmp_path, workers=2, checkpoint_name="checkpoint")
    checkpoint_dir = tmp_path / "checkpoint"

    assert len(list(checkpoint_dir.glob("batch-*.npz"))) == 4
    assert not list(checkpoint_dir.rglob("*.tsv"))
    assert result.uns["scCompass"]["checkpoint"]["computed"] == 4

    resumed = _run(tmp_path, workers=2, checkpoint_name="checkpoint")
    assert resumed.uns["scCompass"]["checkpoint"]["completed_before"] == 4
    assert resumed.uns["scCompass"]["checkpoint"]["computed"] == 0
    np.testing.assert_allclose(
        result.obsm["scCompass_reactions"],
        resumed.obsm["scCompass_reactions"],
    )


def test_parallel_workers_collect_in_memory_without_checkpoint(tmp_path):
    result = _run(tmp_path, workers=2)

    assert result.shape[0] == 4
    assert result.uns["scCompass"]["checkpoint"]["enabled"] is False
    assert result.uns["scCompass"]["parallel"]["scheduler"] == "process"
