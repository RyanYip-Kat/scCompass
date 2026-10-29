"""Validation and projection of AnnData expression into the Compass core."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class PreparedExpression:
    """Gene-by-cell input and provenance required by the Compass core."""

    expression: pd.DataFrame
    symbols: tuple[str, ...]
    mapped_symbols: tuple[str, ...]
    missing_model_genes: tuple[str, ...]
    metadata: dict[str, Any]


def _is_anndata(value):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("scCompass requires anndata") from exc
    return isinstance(value, ad.AnnData)


def _as_cell_by_gene_matrix(matrix, n_vars):
    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            raise ValueError("expression matrix must be two-dimensional")
        if matrix.shape[1] != n_vars:
            raise ValueError("expression matrix shape does not match AnnData.var")
        return matrix

    if sparse.issparse(matrix):
        if matrix.ndim != 2 or matrix.shape[1] != n_vars:
            raise ValueError("expression matrix shape does not match AnnData.var")
        return matrix.tocsr()

    # AnnData backed sparse datasets expose ``to_memory``.
    if hasattr(matrix, "to_memory"):
        return _as_cell_by_gene_matrix(matrix.to_memory(), n_vars)

    raise TypeError(
        "AnnData expression must be a NumPy array or SciPy sparse matrix; "
        f"received {type(matrix).__name__}"
    )


def _to_sparse_frame(matrix, index, columns):
    if sparse.issparse(matrix):
        return pd.DataFrame.sparse.from_spmatrix(
            matrix.tocsr(), index=index, columns=columns
        )
    return pd.DataFrame(matrix, index=index, columns=columns)


def prepare_expression(
    adata,
    expression_layer="counts",
    gene_symbol_key=None,
    model_genes=None,
    sample_size=None,
) -> PreparedExpression:
    """Prepare a sparse gene-by-cell DataFrame for the legacy Compass core.

    Only model-relevant genes are sliced. Cell order is never changed.
    Duplicate gene symbols are summed, matching the legacy preprocessing path.
    """

    if not _is_anndata(adata):
        raise TypeError("adata must be an anndata.AnnData object")
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData.obs_names must be unique")
    input_cells = int(adata.n_obs)
    input_genes = int(adata.n_vars)

    if expression_layer is None:
        layer_label = "X"
    elif expression_layer == "counts":
        if "counts" not in adata.layers:
            raise ValueError(
                "expression layer 'counts' is not present; "
                "use expression_layer=None to select X"
            )
        layer_label = "counts"
    else:
        if expression_layer not in adata.layers:
            raise ValueError(
                f"expression layer {expression_layer!r} is missing; "
                f"available layers: {list(adata.layers)}"
            )
        layer_label = expression_layer

    if gene_symbol_key is None:
        raw_symbols = pd.Index(adata.var_names, dtype="object")
    else:
        if gene_symbol_key not in adata.var:
            raise ValueError(
                f"gene_symbol_key {gene_symbol_key!r} is missing from AnnData.var"
            )
        raw_symbols = pd.Index(adata.var[gene_symbol_key], dtype="object")
    symbols = raw_symbols.astype(str).str.upper()

    all_symbols = tuple(sorted(set(symbols)))
    if model_genes is None:
        selected_symbols = set(symbols)
        missing = ()
    else:
        selected_symbols = {str(x).upper() for x in model_genes}
        missing = tuple(sorted(selected_symbols.difference(all_symbols)))
        selected_symbols.intersection_update(all_symbols)

    selected_rows = np.flatnonzero(symbols.isin(selected_symbols))
    if sample_size is not None:
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        adata = adata[:sample_size, :]
    cell_names = adata.obs_names

    selected_view = adata[:, selected_rows]
    matrix = selected_view.X if expression_layer is None else selected_view.layers[layer_label]
    matrix = _as_cell_by_gene_matrix(matrix, selected_view.n_vars)

    if selected_rows.size:
        selected_names = symbols.take(selected_rows)
        gene_by_cell = _to_sparse_frame(
            matrix.T, index=selected_names, columns=cell_names
        )
        expression = gene_by_cell.groupby(level=0, sort=True).sum()
    else:
        expression = _to_sparse_frame(
            matrix[:, :0].T,
            index=pd.Index([], name=None),
            columns=cell_names,
        )

    expression.index = pd.Index(expression.index.astype(str), name="GeneSymbol")
    expression.columns = pd.Index(expression.columns.astype(str), name=None)
    mapped = tuple(expression.index)

    return PreparedExpression(
        expression=expression,
        symbols=all_symbols,
        mapped_symbols=mapped,
        missing_model_genes=missing,
        metadata={
            "expression_layer": layer_label,
            "gene_symbol_key": gene_symbol_key,
            "input_cells": input_cells,
            "selected_cells": int(len(cell_names)),
            "input_genes": input_genes,
            "selected_model_genes": int(len(selected_symbols)),
            "mapped_model_genes": int(len(mapped)),
            "missing_model_genes": int(len(missing)),
        },
    )
