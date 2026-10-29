"""Construction of an input-preserving scCompass result AnnData."""

from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd


def _cells_by_features(frame, cell_names, name):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{name} must have unique cell and feature identifiers")

    expected = pd.Index(cell_names.astype(str))
    if len(expected.intersection(frame.index)) == len(expected):
        ordered = frame.reindex(expected)
    elif len(expected.intersection(frame.columns)) == len(expected):
        ordered = frame.reindex(columns=expected).T
    else:
        raise ValueError(f"{name} must use the same cells as AnnData.obs_names")

    if ordered.isna().any().any():
        raise ValueError(f"{name} is missing values for some AnnData cells")
    ordered.index = expected
    return ordered


def _copy_mapping(mapping):
    copied = {}
    for key, value in mapping.items():
        copied[key] = value.copy() if hasattr(value, "copy") else deepcopy(value)
    return copied


def build_result(
    source,
    reaction_scores,
    reaction_metadata=None,
    uptake=None,
    secretion=None,
    run_metadata=None,
):
    """Add Compass results to a complete copy of the source AnnData.

    ``source`` is the input AnnData. Reaction and metabolite tables may use
    either cells-by-features or legacy features-by-cells orientation.
    """

    scores = _cells_by_features(reaction_scores, source.obs_names, "reaction_scores")
    result = source.to_memory() if getattr(source, "isbacked", False) else source.copy()
    result.obsm["scCompass_reactions"] = scores.to_numpy(dtype=np.float64)
    result.uns["scCompass_reaction_columns"] = scores.columns.astype(str).to_numpy()

    if reaction_metadata is not None:
        metadata = reaction_metadata.reindex(scores.columns)
        if metadata.index.has_duplicates:
            raise ValueError("reaction_metadata must have unique reaction IDs")
    else:
        metadata = pd.DataFrame(index=scores.columns.copy())
    result.uns["scCompass_reaction_var"] = metadata

    outputs = {"reactions": "scCompass_reactions"}
    if uptake is not None:
        uptake_frame = _cells_by_features(uptake, source.obs_names, "uptake")
        result.obsm["scCompass_uptake"] = uptake_frame.to_numpy(dtype=np.float64)
        result.uns["scCompass_uptake_columns"] = uptake_frame.columns.to_numpy()
        outputs["uptake"] = "scCompass_uptake"
    if secretion is not None:
        secretion_frame = _cells_by_features(
            secretion, source.obs_names, "secretion"
        )
        result.obsm["scCompass_secretion"] = secretion_frame.to_numpy(dtype=np.float64)
        result.uns["scCompass_secretion_columns"] = secretion_frame.columns.to_numpy()
        outputs["secretion"] = "scCompass_secretion"

    metadata = dict(run_metadata or {})
    metadata["outputs"] = outputs
    result.uns["scCompass"] = metadata
    return result


def to_score_anndata(result):
    """Create the legacy cells-by-reactions view from a preserved result."""

    from .io import to_dataframe

    scores = to_dataframe(result, key="reactions")
    output = ad.AnnData(X=scores.to_numpy(dtype=np.float64), obs=result.obs.copy())
    output.var_names = scores.columns.copy()
    reaction_metadata = result.uns.get("scCompass_reaction_var")
    if reaction_metadata is not None:
        metadata = reaction_metadata.reindex(output.var_names)
        for column in metadata.columns:
            output.var[column] = metadata[column].to_numpy()
    output.obsm = _copy_mapping(result.obsm)
    output.obsp = _copy_mapping(result.obsp)
    output.uns = _copy_mapping(result.uns)
    output.obsm.pop("scCompass_reactions", None)
    return output
