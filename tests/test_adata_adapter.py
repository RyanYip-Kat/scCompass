import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scCompass.adata import prepare_expression


def test_adapter_prefers_counts_and_keeps_cell_order():
    matrix = sparse.csr_matrix([[1, 2], [3, 4]])
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.log1p(matrix.toarray())),
        layers={"counts": matrix},
    )
    adata.obs_names = ["cell_b", "cell_a"]
    adata.var_names = ["g1", "g2"]

    prepared = prepare_expression(
        adata,
        expression_layer="counts",
        model_genes={"G1"},
        sample_size=1,
    )

    assert prepared.expression.shape == (1, 1)
    assert prepared.expression.index.tolist() == ["G1"]
    assert prepared.expression.columns.tolist() == ["cell_b"]
    assert prepared.expression.iloc[0, 0] == 1
    assert prepared.metadata["expression_layer"] == "counts"


def test_adapter_sums_duplicate_symbols_without_reordering_cells():
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            [
                [1, 10, 100, 1000],
                [2, 20, 200, 2000],
                [3, 30, 300, 3000],
                [4, 40, 400, 4000],
            ]
        ),
        var={"symbol": ["A", "A", "B", "B"]},
    )
    adata.obs_names = ["b1", "a1", "b2", "a2"]

    prepared = prepare_expression(
        adata,
        expression_layer=None,
        gene_symbol_key="symbol",
        model_genes={"B"},
    )

    assert prepared.expression.shape == (1, 4)
    assert prepared.expression.index.tolist() == ["B"]
    assert prepared.expression.columns.tolist() == ["b1", "a1", "b2", "a2"]
    assert prepared.expression.loc["B"].tolist() == [1100, 2200, 3300, 4400]


def test_adapter_rejects_unknown_layer():
    adata = ad.AnnData(X=sparse.csr_matrix([[1.0]]))

    with pytest.raises(ValueError, match="expression layer.*missing"):
        prepare_expression(
            adata,
            expression_layer="missing",
            model_genes={"G1"},
        )
