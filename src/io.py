"""AnnData persistence and tabular views for scCompass results."""

from pathlib import Path

import pandas as pd


def to_dataframe(result, key="reactions"):
    """Return a cell-by-feature DataFrame from a scCompass result."""

    if key == "reactions":
        if "scCompass_reactions" in result.obsm:
            columns = result.uns.get("scCompass_reaction_columns")
            if columns is None:
                columns = result.uns.get("scCompass_reaction_var", pd.DataFrame()).index
            return pd.DataFrame(
                result.obsm["scCompass_reactions"],
                index=result.obs_names.copy(),
                columns=columns,
            )
        return pd.DataFrame(
            result.X.copy(),
            index=result.obs_names.copy(),
            columns=result.var_names.copy(),
        )
    obsm_key = f"scCompass_{key}"
    if obsm_key not in result.obsm:
        raise KeyError(f"result does not contain {key!r} values")
    columns_key = f"{obsm_key}_columns"
    columns = result.uns.get(columns_key)
    if columns is None:
        columns = [f"{key}_{i}" for i in range(result.obsm[obsm_key].shape[1])]
    return pd.DataFrame(
        result.obsm[obsm_key],
        index=result.obs_names.copy(),
        columns=columns,
    )


def export_table(result, path, key="reactions", sep="\t", **kwargs):
    """Write a result matrix as a tabular file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(result, key=key).to_csv(destination, sep=sep, **kwargs)
    return destination


def save_h5ad(result, path, compression="gzip"):
    """Persist the result AnnData and return the written path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination, compression=compression)
    return destination
