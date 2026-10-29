import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import scCompass
from scCompass.core.compass.microclustering import microcluster
from scCompass.microcluster import pool_expression, unpool_table


def test_pool_expression_and_unpool_table_preserve_cell_mapping():
    expression = pd.DataFrame(
        [[1.0, 10.0, 3.0, 30.0], [2.0, 20.0, 4.0, 40.0]],
        index=["G1", "G2"],
        columns=["c1", "c2", "c3", "c4"],
    )
    pools = {0: [0, 2], 1: [1, 3]}

    pooled = pool_expression(expression, pools)
    unpooled = unpool_table(
        pd.DataFrame(
            [[5.0, 6.0], [7.0, 8.0]],
            index=["R1", "R2"],
            columns=pooled.columns,
        ),
        pools,
        expression.columns,
    )

    np.testing.assert_allclose(pooled.to_numpy(), [[2.0, 20.0], [3.0, 30.0]])
    assert unpooled.columns.tolist() == ["c1", "c2", "c3", "c4"]
    np.testing.assert_allclose(unpooled.to_numpy(), [[5.0, 6.0, 5.0, 6.0], [7.0, 8.0, 7.0, 8.0]])


def test_legacy_microcluster_runs_with_current_pandas():
    rng = np.random.default_rng(7)
    expression = pd.DataFrame(
        rng.random((20, 8)) * 10,
        index=[f"G{i}" for i in range(20)],
        columns=[f"c{i}" for i in range(8)],
    )

    pools = microcluster(
        expression,
        cellsPerPartition=2,
        filterInput="novar",
        n_jobs=1,
    )

    assigned = sorted(index for values in pools.values() for index in values)
    assert assigned == list(range(8))


def test_runner_applies_microclusters_and_expands_scores():
    matrix = sparse.csr_matrix(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    )
    adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix.copy()},
        obs=pd.DataFrame(index=["c1", "c2", "c3", "c4"]),
    )
    adata.var_names = ["G1", "G2"]
    model = type(
        "Model",
        (),
        {
            "name": "micro-model",
            "reactions": {
                "R1": type("Reaction", (), {"list_genes": lambda self: {"G1"}})()
            },
        },
    )()
    pools = {0: [0, 2], 1: [1, 3]}
    captured = {}

    def core_runner(expression, model, media, species, args):
        captured["columns"] = expression.columns.tolist()
        return {
            "reactions": pd.DataFrame(
                [[0.1, 0.2]],
                index=["R1"],
                columns=expression.columns,
            )
        }

    result = scCompass.compass(
        adata,
        model=model.name,
        expression_layer="counts",
        model_loader=lambda name, species: model,
        core_runner=core_runner,
        microcluster_size=2,
        microcluster_pooler=lambda expression, size: pools,
    )

    assert captured["columns"] == [0, 1]
    np.testing.assert_allclose(
        result.obsm["scCompass_reactions"][:, 0],
        [0.1, 0.2, 0.1, 0.2],
    )
    assert result.obs["scCompass_microcluster"].tolist() == [0, 1, 0, 1]
    assert result.uns["scCompass"]["microcluster"]["clusters"] == 2
    assert result.uns["scCompass"]["microcluster"]["cells"] == 4
