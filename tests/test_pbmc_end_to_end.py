from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib.figure import Figure
from scipy import sparse

import scCompass
import scCompass.pl
from scCompass.io import export_table
from scCompass.core.models.MetabolicModel import (
    Association,
    Gene,
    MetabolicModel,
    Reaction,
    Species,
)


PBMC = Path("testdata/pbmc5k_querydata.h5ad")


def _tiny_metabolic_model(genes):
    metabolites = [Species() for _ in range(3)]
    for metabolite, name in zip(metabolites, ["M1", "M2", "M3"]):
        metabolite.id = name

    inbound = Reaction()
    inbound.id, inbound.lower_bound, inbound.upper_bound = "EX_IN", 0.0, 10.0
    inbound.products["M1"] = 1.0

    linked = []
    for index, gene_name in enumerate(genes):
        source = "M1" if index == 0 else "M2"
        target = "M2" if index == 0 else "M3"
        reaction = Reaction()
        reaction.id = f"R_GENE_{index}"
        reaction.lower_bound, reaction.upper_bound = 0.0, 10.0
        reaction.reactants[source] = 1.0
        reaction.products[target] = 1.0
        gene = Gene()
        gene.id = gene.name = gene_name
        association = Association()
        association.type, association.gene = "gene", gene
        reaction.gene_associations = association
        linked.append(reaction)

    outbound = Reaction()
    outbound.id, outbound.lower_bound, outbound.upper_bound = "EX_OUT", 0.0, 10.0
    outbound.reactants["M3"] = 1.0

    model = MetabolicModel("PBMCRealCoreModel")
    model.reactions = {
        reaction.id: reaction
        for reaction in [inbound, *linked, outbound]
    }
    model.species = {metabolite.id: metabolite for metabolite in metabolites}
    model.limitExchangeReactions(1.0)
    model.make_unidirectional()
    return model


def test_pbmc_bounded_end_to_end_h5ad_table_and_plot(tmp_path):
    adata = ad.read_h5ad(PBMC, backed="r")
    model_genes = set(adata.var_names[:5])
    reaction = SimpleNamespace(list_genes=lambda: model_genes)
    model = SimpleNamespace(
        name="PBMC_TEST_MODEL",
        reactions={"R_TEST": reaction},
        media="default-media",
    )

    def core_runner(expression, model, media, species, args):
        scores = pd.DataFrame(
            [range(expression.shape[1])],
            index=["R_TEST"],
            columns=expression.columns,
            dtype=float,
        )
        return {"reactions": scores}

    result = scCompass.compass(
        adata,
        model="PBMC_TEST_MODEL",
        expression_layer="counts",
        sample_size=8,
        model_loader=lambda name, species: model,
        core_runner=core_runner,
    )

    assert result.shape == (8, adata.n_vars)
    assert result.obs_names.tolist() == adata.obs_names[:8].tolist()
    assert "X_umap" in result.obsm
    assert result.uns["scCompass"]["expression_layer"] == "counts"
    assert result.uns["scCompass"]["input"]["input_cells"] == adata.n_obs

    result_path = tmp_path / "pbmc_scCompass.h5ad"
    result.write_h5ad(result_path, compression="gzip")
    restored = ad.read_h5ad(result_path)
    assert restored.shape == result.shape
    assert restored.var_names.tolist() == adata.var_names.tolist()
    assert restored.uns["scCompass_reaction_columns"].tolist() == ["R_TEST"]

    table_path = export_table(restored, tmp_path / "reactions.tsv")
    assert pd.read_csv(table_path, sep="\t", index_col=0).shape == (8, 1)

    figure = scCompass.pl.umap(restored, color=["R_TEST"])
    assert isinstance(figure, Figure)


def test_pbmc_bounded_data_runs_real_compass_core(tmp_path, monkeypatch):
    from scCompass.core.compass import cache

    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(tmp_path / "cache"))
    cache._cache.clear()
    cache._new_cache.clear()

    adata = ad.read_h5ad(PBMC, backed="r")
    model = _tiny_metabolic_model(["GAPDH", "LDHA"])
    result = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        sample_size=8,
        lambda_=0.0,
        model_loader=lambda name, species: model,
        checkpoint_dir=tmp_path / "checkpoint",
        checkpoint_batch_size=4,
        num_processes=2,
        num_threads=1,
    )

    assert result.shape[0] == 8
    assert "R_GENE_0_pos" in result.uns["scCompass_reaction_columns"]
    assert "R_GENE_1_pos" in result.uns["scCompass_reaction_columns"]
    assert result.obsm["scCompass_reactions"].size > 0
    assert len(list((tmp_path / "checkpoint").glob("batch-*.npz"))) == 2
    assert not list((tmp_path / "checkpoint").rglob("*.tsv"))


def test_pbmc_microclusters_with_parallel_core_and_checkpoint(tmp_path, monkeypatch):
    from scCompass.core.compass import cache

    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(tmp_path / "cache"))
    cache._cache.clear()
    cache._new_cache.clear()

    adata = ad.read_h5ad(PBMC, backed="r")
    model = _tiny_metabolic_model(["GAPDH", "LDHA"])
    checkpoint_dir = tmp_path / "micro_checkpoint"
    result = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        sample_size=8,
        lambda_=0.0,
        model_loader=lambda name, species: model,
        microcluster_size=2,
        microcluster_pooler=lambda expression, size: {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
        },
        checkpoint_dir=checkpoint_dir,
        checkpoint_batch_size=1,
        num_processes=2,
        num_threads=1,
    )

    assert result.shape[0] == 8
    assert result.obs["scCompass_microcluster"].notna().all()
    assert result.uns["scCompass"]["microcluster"]["enabled"] is True
    assert result.uns["scCompass"]["microcluster"]["cells"] == 8
    assert list(checkpoint_dir.glob("batch-*.npz"))
