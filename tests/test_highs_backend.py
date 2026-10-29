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


def _model():
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

    model = MetabolicModel("HighsTestModel")
    model.reactions = {
        reaction.id: reaction
        for reaction in [inbound, *linked, outbound]
    }
    model.species = {metabolite.id: metabolite for metabolite in [m1, m2, m3]}
    model.limitExchangeReactions(1.0)
    model.make_unidirectional()
    return model


def _adata():
    matrix = sparse.csr_matrix([[2.0, 1.0], [1.0, 2.0]])
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["c1", "c2"]),
    )
    adata.var_names = ["G1", "G2"]
    return adata


def _run(tmp_path, backend, workers=1, checkpoint=None):
    from scCompass.core.compass import cache

    cache.PREPROCESS_CACHE_DIR = str(tmp_path / backend)
    cache._cache.clear()
    cache._new_cache.clear()
    model = _model()
    return scCompass.compass(
        _adata(),
        model=model.name,
        expression_layer="counts",
        lambda_=0.0,
        calc_metabolites=True,
        model_loader=lambda name, species: model,
        solver_backend=backend,
        num_processes=workers,
        checkpoint_dir=checkpoint,
        checkpoint_batch_size=1,
    )


def test_highs_backend_matches_cplex_scores_and_exchange(tmp_path):
    cplex_result = _run(tmp_path, "cplex")
    highs_result = _run(tmp_path, "highs")

    np.testing.assert_allclose(
        highs_result.obsm["scCompass_reactions"],
        cplex_result.obsm["scCompass_reactions"],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        highs_result.obsm["scCompass_uptake"],
        cplex_result.obsm["scCompass_uptake"],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        highs_result.obsm["scCompass_secretion"],
        cplex_result.obsm["scCompass_secretion"],
        rtol=0,
        atol=1e-7,
    )
    assert highs_result.uns["scCompass"]["solver_backend"] == "highs"


def test_highs_backend_supports_parallel_workers_and_checkpoints(tmp_path):
    serial = _run(tmp_path, "highs", workers=1)
    parallel_checkpoint = tmp_path / "highs-checkpoint"
    parallel = _run(
        tmp_path,
        "highs",
        workers=2,
        checkpoint=parallel_checkpoint,
    )

    np.testing.assert_allclose(
        parallel.obsm["scCompass_reactions"],
        serial.obsm["scCompass_reactions"],
        rtol=0,
        atol=1e-9,
    )
    assert parallel.uns["scCompass"]["parallel"]["workers"] == 2
    assert len(list(parallel_checkpoint.glob("batch-*.npz"))) == 2
