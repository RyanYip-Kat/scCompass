"""Adapter from in-memory AnnData inputs to the original Compass functions."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import pandas as pd

from .checkpoint import BatchCheckpoint, CellResult, compute_checkpoint_fingerprint
from .exceptions import SolverUnavailableError


_WORKER_STATE = {}


def translate_solver_error(error):
    """Raise an actionable SolverUnavailableError for known CPLEX failures."""

    message = str(error)
    lowered = message.lower()
    if "community edition" in lowered or "problem size limits" in lowered:
        raise SolverUnavailableError(
            "The installed CPLEX Community Edition cannot solve this model "
            "because its problem-size limit was exceeded. Install a full "
            "CPLEX license or run a reduced reaction/model subset."
        ) from error
    if "license" in lowered and "cplex" in lowered:
        raise SolverUnavailableError(
            "CPLEX could not start because no valid license is available."
        ) from error


def _require_cplex():
    try:
        import cplex
    except ImportError as exc:
        raise SolverUnavailableError(
            "Running Compass requires CPLEX and a valid CPLEX license. "
            "Install the IBM CPLEX Python API before using the default core."
        ) from exc
    return cplex


def _solve_one_cell(cell_penalties, model, args):
    if args.get("solver_backend", "cplex") == "highs":
        from .highs_backend import solve_cell

        return solve_cell(model, cell_penalties, args)

    from .core.compass.algorithm import (
        compass_exchange,
        compass_reactions,
        initialize_cplex_problem,
    )

    problem = initialize_cplex_problem(
        model,
        args.get("num_threads", 1),
        args.get("lpmethod", 4),
        args.get("advance", 2),
    )
    try:
        reactions = compass_reactions(
            model,
            problem,
            cell_penalties,
            perf_log=None,
            args=args,
        )
        uptake, secretion, exchange = compass_exchange(
            model,
            problem,
            cell_penalties,
            only_exchange=not args["calc_metabolites"],
            perf_log=None,
            args=args,
        )
        reactions.update(exchange)
        return reactions, uptake, secretion
    finally:
        problem.end()


def _init_parallel_worker(model, args):
    if args.get("solver_backend", "cplex") == "cplex":
        _require_cplex()
    checkpoint = None
    if args.get("checkpoint_dir"):
        checkpoint = BatchCheckpoint(
            args["checkpoint_dir"],
            fingerprint=args["checkpoint_fingerprint"],
            batch_size=args.get("checkpoint_batch_size", 64),
            overwrite=False,
        )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "model": model,
            "args": args,
            "checkpoint": checkpoint,
        }
    )


def _solve_parallel_chunk(tasks):
    model = _WORKER_STATE["model"]
    args = _WORKER_STATE["args"]
    checkpoint = _WORKER_STATE["checkpoint"]
    completed = []
    results = []
    pending = []

    for cell, cell_penalties in tasks:
        try:
            reactions, uptake, secretion = _solve_one_cell(
                cell_penalties, model, args
            )
        except Exception as exc:
            translate_solver_error(exc)
            raise
        completed.append(str(cell))
        if checkpoint is not None:
            pending.append(
                CellResult(
                    cell=str(cell),
                    reactions=pd.Series(reactions, dtype=float),
                    uptake=pd.Series(uptake, dtype=float)
                    if args["calc_metabolites"]
                    else None,
                    secretion=pd.Series(secretion, dtype=float)
                    if args["calc_metabolites"]
                    else None,
                )
            )
            if len(pending) == checkpoint.batch_size:
                checkpoint.write_batch(pending)
                pending = []
        else:
            results.append(
                CellResult(
                    cell=str(cell),
                    reactions=pd.Series(reactions, dtype=float),
                    uptake=pd.Series(uptake, dtype=float)
                    if args["calc_metabolites"]
                    else None,
                    secretion=pd.Series(secretion, dtype=float)
                    if args["calc_metabolites"]
                    else None,
                )
            )
    if pending:
        checkpoint.write_batch(pending)
    return {"cells": completed, "results": results}


def _partition_tasks(cells, penalties, workers):
    indices = list(range(len(cells)))
    chunks = [indices[index::workers] for index in range(workers)]
    return [
        [(cells[index], penalties.iloc[:, index]) for index in chunk]
        for chunk in chunks
        if chunk
    ]


def _parallel_results(
    missing_cells,
    cell_order,
    penalties,
    model,
    args,
    checkpoint,
    completed_before,
):
    workers = min(int(args.get("num_processes", 1)), len(missing_cells))
    if workers <= 1:
        return None

    chunks = _partition_tasks(missing_cells, penalties.loc[:, missing_cells], workers)
    collected = []
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_parallel_worker,
            initargs=(model, args),
        ) as executor:
            futures = [executor.submit(_solve_parallel_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                collected.extend(future.result()["results"])
    except Exception as exc:
        translate_solver_error(exc)
        raise

    if checkpoint is not None:
        frames = checkpoint.load_frames(cell_order)
        reactions = frames["reactions"].T
        uptake = frames["uptake"].T if args["calc_metabolites"] else None
        secretion = frames["secretion"].T if args["calc_metabolites"] else None
    else:
        reactions = pd.DataFrame.from_dict(
            {result.cell: result.reactions for result in collected},
            orient="index",
        ).T.sort_index()
        uptake = (
            pd.DataFrame.from_dict(
                {result.cell: result.uptake for result in collected},
                orient="index",
            ).T.sort_index()
            if args["calc_metabolites"]
            else None
        )
        secretion = (
            pd.DataFrame.from_dict(
                {result.cell: result.secretion for result in collected},
                orient="index",
            ).T.sort_index()
            if args["calc_metabolites"]
            else None
        )
    return {
        "reactions": reactions,
        "uptake": uptake,
        "secretion": secretion,
        "parallel": {
            "workers": workers,
            "tasks": len(missing_cells),
            "scheduler": "process",
        },
        "checkpoint": {
            "enabled": checkpoint is not None,
            "directory": args.get("checkpoint_dir"),
            "fingerprint": args.get("checkpoint_fingerprint"),
            "completed_before": completed_before,
            "computed": len(missing_cells),
            "batch_size": args.get("checkpoint_batch_size", 64),
        },
    }


def run_legacy_core(expression, model, media, species, args):
    """Run the original penalty and FBA functions on an in-memory matrix."""

    checkpoint = None
    completed_before = 0
    missing_cells = list(expression.columns)
    if args.get("checkpoint_dir"):
        fingerprint = args.get("checkpoint_fingerprint")
        if not fingerprint:
            fingerprint = compute_checkpoint_fingerprint(expression, model, args)
        checkpoint = BatchCheckpoint(
            args["checkpoint_dir"],
            fingerprint=fingerprint,
            batch_size=args.get("checkpoint_batch_size", 64),
            overwrite=args.get("overwrite_checkpoint", False),
        )
        completed = checkpoint.completed_cells()
        completed_before = len(completed)
        missing_cells = [cell for cell in expression.columns if cell not in completed]
        if not missing_cells:
            frames = checkpoint.load_frames(expression.columns)
            return {
                "reactions": frames["reactions"].T,
                "uptake": (
                    frames["uptake"].T if args["calc_metabolites"] else None
                ),
                "secretion": (
                    frames["secretion"].T if args["calc_metabolites"] else None
                ),
                "checkpoint": {
                    "enabled": True,
                    "directory": str(args["checkpoint_dir"]),
                    "fingerprint": fingerprint,
                    "completed_before": completed_before,
                    "computed": 0,
                    "batch_size": args.get("checkpoint_batch_size", 64),
                },
                "parallel": {
                    "workers": 0,
                    "tasks": 0,
                    "scheduler": "none",
                },
            }

    if args.get("solver_backend", "cplex") == "cplex":
        _require_cplex()

    backend = args.get("penalty_backend", "vectorized")
    if backend == "vectorized":
        from .penalties import eval_reaction_penalties_vectorized

        penalties = eval_reaction_penalties_vectorized(
            model,
            expression,
            lambda_=args["lambda"],
            num_neighbors=args["num_neighbors"],
            symmetric_kernel=args["symmetric_kernel"],
            and_function=args["and_function"],
            penalty_diffusion_mode=args["penalty_diffusion"],
            input_weights=None,
            input_knn=args.get("input_knn"),
            output_knn=args.get("output_knn"),
            latent_input=args.get("latent_space"),
            cell_chunk_size=args.get("cell_chunk_size", 1024),
        )
    elif backend == "legacy":
        from .core.compass.penalties import eval_reaction_penalties_shared

        penalties = eval_reaction_penalties_shared(
            model,
            expression,
            args["lambda"],
            num_neighbors=args["num_neighbors"],
            symmetric_kernel=args["symmetric_kernel"],
            and_function=args["and_function"],
            penalty_diffusion_mode=args["penalty_diffusion"],
            input_weights=None,
            input_knn=args.get("input_knn"),
            output_knn=args.get("output_knn"),
            latent_input=args.get("latent_space"),
        )
    else:
        raise ValueError(
            "penalty_backend must be 'vectorized' or 'legacy', "
            f"received {backend!r}"
        )

    if int(args.get("num_processes", 1)) > 1 and len(missing_cells) > 1:
        parallel = _parallel_results(
            missing_cells,
            expression.columns,
            penalties,
            model,
            args,
            checkpoint,
            completed_before,
        )
        if parallel is not None:
            return parallel

    reaction_by_cell = {}
    uptake_by_cell = {}
    secretion_by_cell = {}
    pending = []
    for cell in missing_cells:
        cell_penalties = penalties[cell]
        try:
            reactions, uptake, secretion = _solve_one_cell(
                cell_penalties, model, args
            )
        except Exception as exc:
            translate_solver_error(exc)
            raise
        if checkpoint is not None:
            pending.append(
                CellResult(
                    cell=str(cell),
                    reactions=pd.Series(reactions, dtype=float),
                    uptake=pd.Series(uptake, dtype=float)
                    if args["calc_metabolites"]
                    else None,
                    secretion=pd.Series(secretion, dtype=float)
                    if args["calc_metabolites"]
                    else None,
                )
            )
            if len(pending) == checkpoint.batch_size:
                checkpoint.write_batch(pending)
                pending = []
        else:
            reaction_by_cell[cell] = reactions
            if args["calc_metabolites"]:
                uptake_by_cell[cell] = uptake
                secretion_by_cell[cell] = secretion

    if checkpoint is not None:
        if pending:
            checkpoint.write_batch(pending)
        frames = checkpoint.load_frames(expression.columns)
        return {
            "reactions": frames["reactions"].T,
            "uptake": frames["uptake"].T if args["calc_metabolites"] else None,
            "secretion": frames["secretion"].T if args["calc_metabolites"] else None,
            "checkpoint": {
                "enabled": True,
                "directory": str(args["checkpoint_dir"]),
                "fingerprint": checkpoint.fingerprint,
                "completed_before": completed_before,
                "computed": len(missing_cells),
                "batch_size": checkpoint.batch_size,
            },
            "parallel": {
                "workers": 1,
                "tasks": len(missing_cells),
                "scheduler": "serial",
            },
        }

    def to_legacy_frame(values):
        if not values:
            return None
        return pd.DataFrame.from_dict(values, orient="index").T.sort_index()

    return {
        "reactions": to_legacy_frame(reaction_by_cell),
        "uptake": to_legacy_frame(uptake_by_cell),
        "secretion": to_legacy_frame(secretion_by_cell),
        "parallel": {
            "workers": 1,
            "tasks": len(missing_cells),
            "scheduler": "serial",
        },
    }
