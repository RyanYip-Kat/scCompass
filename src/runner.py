"""End-to-end orchestration from AnnData to an scCompass result."""

from collections.abc import Mapping

import pandas as pd

from ._version import __version__
from .adata import prepare_expression
from .checkpoint import compute_checkpoint_fingerprint
from .microcluster import (
    default_pooler,
    pool_expression,
    unpool_table,
    validate_pools,
)
from .result import build_result


def _default_model_loader(model, species, media, isoform_summing):
    from .core import models
    from .core.globals import EXCHANGE_LIMIT

    return models.init_model(
        model=model,
        species=species,
        exchange_limit=EXCHANGE_LIMIT,
        media=media,
        isoform_summing=isoform_summing,
    )


def _default_core_runner(expression, model, media, species, args):
    from ._compat_core import run_legacy_core

    return run_legacy_core(expression, model, media, species, args)


def _model_gene_symbols(model):
    genes = set()
    for reaction in model.reactions.values():
        genes.update(reaction.list_genes())
    return genes


def _reaction_metadata(model):
    rows = {}
    for reaction_id, reaction in model.reactions.items():
        reverse = getattr(reaction, "reverse_reaction", None)
        rows[reaction_id] = {
            "reaction_name": getattr(reaction, "name", ""),
            "subsystem": getattr(reaction, "subsystem", ""),
            "lower_bound": getattr(reaction, "lower_bound", float("nan")),
            "upper_bound": getattr(reaction, "upper_bound", float("nan")),
            "is_exchange": bool(getattr(reaction, "is_exchange", False)),
            "reverse_reaction": getattr(reverse, "id", "") if reverse else "",
            "genes": ";".join(sorted(reaction.list_genes())),
            "reactants": ";".join(
                sorted(getattr(reaction, "reactants", {}))
            ),
            "products": ";".join(
                sorted(getattr(reaction, "products", {}))
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _validate_core_output(output):
    if not isinstance(output, Mapping):
        raise TypeError("core_runner must return a mapping of result tables")
    if "reactions" not in output or output["reactions"] is None:
        raise ValueError("core_runner did not return reaction scores")
    return output


def run_compass(
    adata,
    model="RECON2_mat",
    media="default-media",
    species="homo_sapiens",
    expression_layer="counts",
    gene_symbol_key=None,
    lambda_=0.0,
    num_neighbors=30,
    calc_metabolites=True,
    isoform_summing="remove-summing",
    and_function="mean",
    symmetric_kernel=False,
    penalty_diffusion="knn",
    penalty_backend="vectorized",
    cell_chunk_size=1024,
    checkpoint_dir=None,
    checkpoint_batch_size=64,
    overwrite_checkpoint=False,
    num_processes=1,
    microcluster_size=None,
    microcluster_pooler=None,
    microcluster_filter="fano",
    microcluster_filter_threshold=None,
    solver_backend="cplex",
    select_reactions=None,
    select_subsystems=None,
    test_mode=False,
    num_threads=1,
    model_loader=None,
    core_runner=None,
    sample_size=None,
    **core_options,
):
    """Run Compass and return a score-centric result AnnData."""

    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1")
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive")
    if penalty_backend not in {"vectorized", "legacy"}:
        raise ValueError(
            "penalty_backend must be 'vectorized' or 'legacy'"
        )
    if cell_chunk_size < 1:
        raise ValueError("cell_chunk_size must be positive")
    if checkpoint_batch_size < 1:
        raise ValueError("checkpoint_batch_size must be positive")
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    if microcluster_size is not None and microcluster_size < 1:
        raise ValueError("microcluster_size must be positive")
    if solver_backend not in {"cplex", "highs"}:
        raise ValueError("solver_backend must be 'cplex' or 'highs'")

    result_source = adata if sample_size is None else adata[:sample_size]
    loader = model_loader or _default_model_loader
    if model_loader is None:
        model_object = loader(model, species, media, isoform_summing)
    else:
        model_object = loader(model, species)

    prepared = prepare_expression(
        adata,
        expression_layer=expression_layer,
        gene_symbol_key=gene_symbol_key,
        model_genes=_model_gene_symbols(model_object),
        sample_size=sample_size,
    )
    execution_expression = prepared.expression
    pools = None
    microcluster_metadata = {"enabled": False}
    if microcluster_size is not None:
        pooler = microcluster_pooler or default_pooler
        if microcluster_pooler is None:
            pools = pooler(
                prepared.expression,
                microcluster_size,
                n_jobs=num_processes,
                filter_input=microcluster_filter,
                filter_threshold=microcluster_filter_threshold,
            )
        else:
            pools = pooler(prepared.expression, microcluster_size)
        labels = validate_pools(pools, prepared.expression.columns)
        execution_expression = pool_expression(prepared.expression, pools)
        microcluster_metadata = {
            "enabled": True,
            "cells": int(prepared.expression.shape[1]),
            "clusters": int(execution_expression.shape[1]),
            "target_cells_per_cluster": int(microcluster_size),
            "filter": microcluster_filter,
            "filter_threshold": microcluster_filter_threshold,
            "cell_to_cluster": {
                str(prepared.expression.columns[index]): int(cluster)
                for index, cluster in labels.items()
            },
        }
    args = {
        "lambda": float(lambda_),
        "num_neighbors": int(num_neighbors),
        "symmetric_kernel": bool(symmetric_kernel),
        "penalty_diffusion": penalty_diffusion,
        "penalty_backend": penalty_backend,
        "cell_chunk_size": int(cell_chunk_size),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_batch_size": int(checkpoint_batch_size),
        "overwrite_checkpoint": bool(overwrite_checkpoint),
        "num_processes": int(num_processes),
        "microcluster_size": microcluster_size,
        "microcluster_filter": microcluster_filter,
        "solver_backend": solver_backend,
        "input_weights": None,
        "input_knn": None,
        "output_knn": None,
        "latent_space": None,
        "isoform_summing": isoform_summing,
        "and_function": and_function,
        "select_reactions": select_reactions,
        "select_subsystems": select_subsystems,
        "test_mode": bool(test_mode),
        "calc_metabolites": bool(calc_metabolites),
        "no_reactions": False,
        "save_argmaxes": False,
        "num_threads": int(num_threads),
        "lpmethod": 4,
        "advance": 2,
        "generate_cache": False,
        "detailed_perf": False,
    }
    args.update(core_options)
    if checkpoint_dir is not None:
        args["checkpoint_fingerprint"] = compute_checkpoint_fingerprint(
            prepared.expression, model_object, args
        )
    runner = core_runner or _default_core_runner
    output = _validate_core_output(
        runner(execution_expression, model_object, media, species, args)
    )
    if pools is not None:
        output = dict(output)
        output["reactions"] = unpool_table(
            output["reactions"], pools, prepared.expression.columns
        )
        for key in ("uptake", "secretion"):
            if output.get(key) is not None:
                output[key] = unpool_table(
                    output[key], pools, prepared.expression.columns
                )

    run_metadata = {
        "version": __version__,
        "model": model,
        "media": media,
        "species": species,
        "expression_layer": prepared.metadata["expression_layer"],
        "gene_symbol_key": gene_symbol_key,
        "lambda": float(lambda_),
        "num_neighbors": int(num_neighbors),
        "penalty_backend": penalty_backend,
        "cell_chunk_size": int(cell_chunk_size),
        "num_processes": int(num_processes),
        "microcluster_filter": microcluster_filter,
        "microcluster_filter_threshold": microcluster_filter_threshold,
        "solver_backend": solver_backend,
        "calc_metabolites": bool(calc_metabolites),
        "core_args": {
            key: value
            for key, value in args.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "input": prepared.metadata,
        "mapped_model_genes": list(prepared.mapped_symbols),
        "missing_model_genes": list(prepared.missing_model_genes),
        "microcluster": microcluster_metadata,
    }
    run_metadata["checkpoint"] = output.get(
        "checkpoint",
        {"enabled": False},
    )
    run_metadata["parallel"] = output.get(
        "parallel",
        {"workers": 1, "tasks": prepared.metadata["selected_cells"], "scheduler": "serial"},
    )
    result = build_result(
        result_source,
        output["reactions"],
        reaction_metadata=_reaction_metadata(model_object),
        uptake=output.get("uptake"),
        secretion=output.get("secretion"),
        run_metadata=run_metadata,
    )
    if pools is not None:
        result.obs["scCompass_microcluster"] = [
            labels[index] for index in range(result.n_obs)
        ]
    result.uns["scCompass"]["status"] = "success"
    return result
