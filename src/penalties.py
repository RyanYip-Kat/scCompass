"""Vectorized, sparse-aware reaction expression and penalty calculation."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from .core.compass.penalties import (
    sample_weights_knn,
    sample_weights_tsne_symmetric,
)
from .core.globals import PCA_SEED


def _association_spec(association, row_index):
    if association is None:
        return None
    if association.type == "gene":
        gene = association.gene
        primary = row_index.get(gene.name)
        aliases = tuple(
            row_index[symbol]
            for symbol in gene.alt_symbols
            if symbol in row_index
        )
        return ("gene", primary, aliases)
    if association.type in {"and", "or"}:
        return (
            association.type,
            tuple(_association_spec(child, row_index) for child in association.children),
        )
    raise ValueError(f"Unknown gene association type: {association.type!r}")


def _gene_values(spec, matrix):
    _, primary, aliases = spec
    if primary is not None:
        return matrix[primary]
    if aliases:
        values = matrix[list(aliases)]
        return np.mean(values, axis=0)
    return np.full(matrix.shape[1], np.nan, dtype=float)


def _association_values(spec, matrix, and_function):
    if spec is None:
        return np.full(matrix.shape[1], np.nan, dtype=float)
    kind = spec[0]
    if kind == "gene":
        return _gene_values(spec, matrix)
    payload = spec[1]

    values = np.vstack(
        [_association_values(child, matrix, and_function) for child in payload]
    )
    if kind == "or":
        return np.nansum(values, axis=0)
    if and_function == "min":
        result = np.min(values, axis=0)
        result[np.isnan(values).any(axis=0)] = np.nan
        return result

    zero_filled = np.nan_to_num(values, nan=0.0)
    if and_function == "mean":
        return np.mean(zero_filled, axis=0)
    if and_function == "median":
        return np.median(zero_filled, axis=0)
    raise ValueError(f"Invalid value for and_function: {and_function}")


def eval_reaction_expression_vectorized(
    model,
    expression,
    and_function="mean",
    cell_chunk_size=1024,
):
    """Evaluate all legacy GPR reaction-expression values in cell chunks.

    The returned matrix has reactions as rows and cells as columns and includes
    the legacy ``log2(expression + 1)`` transformation.
    """

    if cell_chunk_size < 1:
        raise ValueError("cell_chunk_size must be positive")
    if not expression.index.is_unique:
        raise ValueError("expression gene symbols must be unique")

    row_index = {str(symbol): i for i, symbol in enumerate(expression.index)}
    reactions = list(model.reactions.values())
    specs = [
        _association_spec(reaction.gene_associations, row_index)
        for reaction in reactions
    ]
    result = np.empty((len(reactions), expression.shape[1]), dtype=float)

    for start in range(0, expression.shape[1], cell_chunk_size):
        stop = min(expression.shape[1], start + cell_chunk_size)
        matrix = expression.iloc[:, start:stop].to_numpy(dtype=float)
        for reaction_index, spec in enumerate(specs):
            result[reaction_index, start:stop] = _association_values(
                spec, matrix, and_function
            )

    result = np.where(np.isnan(result), 0.0, result)
    result = np.log2(result + 1.0)
    return pd.DataFrame(
        result,
        index=[reaction.id for reaction in reactions],
        columns=expression.columns.copy(),
    )


def _diffusion_weights(
    expression,
    lambda_,
    num_neighbors,
    symmetric_kernel,
    penalty_diffusion_mode,
    input_weights,
    input_knn,
    output_knn,
    latent_input,
):
    if input_weights is not None:
        if isinstance(input_weights, pd.DataFrame):
            if not input_weights.index.equals(expression.columns) or not input_weights.columns.equals(
                expression.columns
            ):
                raise ValueError("input_weights must match expression cell order")
            return input_weights.to_numpy(dtype=float)
        return np.asarray(input_weights, dtype=float)
    if lambda_ == 0:
        return sparse.csr_matrix(
            (expression.shape[1], expression.shape[1]), dtype=float
        )

    if latent_input is not None:
        data = pd.read_csv(latent_input, sep="\t", index_col=0).T
    else:
        values = np.log2(expression.to_numpy(dtype=float) + 1.0)
        n_components = min(values.shape[0], values.shape[1], 20)
        pca = PCA(n_components=n_components, random_state=PCA_SEED)
        data = pd.DataFrame(
            pca.fit_transform(values.T).T,
            columns=expression.columns,
        )

    if penalty_diffusion_mode == "gaussian":
        weights = sample_weights_tsne_symmetric(
            data, num_neighbors, symmetric_kernel
        )
        return weights.to_numpy(dtype=float)
    if penalty_diffusion_mode == "knn":
        return sample_weights_knn(data, num_neighbors, input_knn, output_knn)
    raise ValueError(
        f"Invalid value for penalty_diffusion_mode: {penalty_diffusion_mode}"
    )


def eval_reaction_penalties_vectorized(
    model,
    expression,
    lambda_,
    num_neighbors,
    symmetric_kernel,
    and_function,
    penalty_diffusion_mode,
    input_weights=None,
    input_knn=None,
    output_knn=None,
    latent_input=None,
    cell_chunk_size=1024,
):
    """Compute legacy-compatible reaction penalties with vectorized GPRs."""

    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1")
    reaction_expression = eval_reaction_expression_vectorized(
        model,
        expression,
        and_function=and_function,
        cell_chunk_size=cell_chunk_size,
    )
    weights = _diffusion_weights(
        expression,
        lambda_=lambda_,
        num_neighbors=num_neighbors,
        symmetric_kernel=symmetric_kernel,
        penalty_diffusion_mode=penalty_diffusion_mode,
        input_weights=input_weights,
        input_knn=input_knn,
        output_knn=output_knn,
        latent_input=latent_input,
    )
    neighborhood = weights.dot(reaction_expression.T).T
    neighborhood = pd.DataFrame(
        neighborhood,
        index=reaction_expression.index,
        columns=reaction_expression.columns,
    )
    result = (1 - lambda_) / (1 + reaction_expression) + lambda_ / (
        1 + neighborhood
    )
    result.index.name = "Reaction"
    return result
