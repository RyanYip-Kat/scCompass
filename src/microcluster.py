"""Microcluster pooling helpers for Compass expression matrices."""

import pandas as pd


def validate_pools(pools, cell_names):
    """Validate a partition and return a cell-index to cluster mapping."""

    labels = {}
    for cluster, indices in pools.items():
        for index in indices:
            index = int(index)
            if index < 0 or index >= len(cell_names):
                raise ValueError(f"Microcluster index {index} is out of range")
            if index in labels:
                raise ValueError(f"Cell index {index} occurs in multiple microclusters")
            labels[index] = int(cluster)
    if set(labels) != set(range(len(cell_names))):
        raise ValueError("Microclusters must partition every input cell")
    return labels


def pool_expression(expression, pools):
    """Average gene expression within validated microclusters."""

    labels = validate_pools(pools, expression.columns)
    by_cell = pd.Series(
        [labels[index] for index in range(expression.shape[1])],
        index=expression.columns,
        name="compass_microcluster",
    )
    pooled = expression.T.groupby(by_cell, sort=True).mean().T
    pooled.columns = pd.Index(pooled.columns, name=None)
    return pooled


def unpool_table(pooled, pools, cell_names):
    """Expand pooled feature values back to original cell order."""

    labels = validate_pools(pools, cell_names)
    columns = []
    for index, cell_name in enumerate(cell_names):
        cluster = labels[index]
        matches = [
            column for column in pooled.columns if str(column) == str(cluster)
        ]
        if not matches:
            raise ValueError(f"Pooled result is missing microcluster {cluster}")
        columns.append(pooled.loc[:, matches[0]].rename(cell_name))
    if not columns:
        return pd.DataFrame(index=pooled.index)
    return pd.concat(columns, axis=1).loc[:, cell_names]


def default_pooler(
    expression,
    cells_per_partition,
    n_jobs=1,
    filter_input="fano",
    filter_threshold=None,
):
    from .core.compass.microclustering import microcluster

    return microcluster(
        expression,
        cellsPerPartition=cells_per_partition,
        filterInput=filter_input,
        filterThreshold=filter_threshold,
        n_jobs=n_jobs,
    )
