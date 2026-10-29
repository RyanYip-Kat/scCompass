#!/usr/bin/env python
"""Run the full PBMC5k scCompass pipeline on every cell in the fixture."""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import scCompass
from scCompass.pipeline import run_pipeline
from scCompass.submodels import build_central_carbon_submodel


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete scCompass pipeline on all cells in "
            "testdata/pbmc5k_querydata.h5ad."
        )
    )
    parser.add_argument(
        "--input",
        default="testdata/pbmc5k_querydata.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pbmc5k_full_pipeline",
    )
    parser.add_argument(
        "--model-scope",
        choices=["central-carbon", "full"],
        default="central-carbon",
        help=(
            "central-carbon is runnable with CPLEX Community; "
            "full is the complete RECON2 model and requires a full CPLEX license"
        ),
    )
    parser.add_argument("--model", default="RECON2_mat")
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument("--media", default="default-media")
    parser.add_argument("--layer", default="counts")
    parser.add_argument("--solver-backend", choices=["cplex", "highs"], default="cplex")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--microcluster-size", type=int, default=1000)
    parser.add_argument(
        "--microcluster-filter",
        choices=["fano", "novar", "threshold"],
        default="novar",
    )
    parser.add_argument("--checkpoint-batch-size", type=int, default=16)
    parser.add_argument("--plot-top-reactions", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--export-full-table", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    options = {
        "model": args.model,
        "media": args.media,
        "species": args.species,
        "expression_layer": None if args.layer == "X" else args.layer,
        "solver_backend": args.solver_backend,
        "num_processes": args.num_processes,
        "num_threads": args.num_threads,
        "microcluster_size": args.microcluster_size,
        "microcluster_filter": args.microcluster_filter,
        "overwrite_checkpoint": args.overwrite_checkpoint,
        "checkpoint_batch_size": args.checkpoint_batch_size,
    }

    runner = None
    if args.model_scope == "central-carbon":
        model = build_central_carbon_submodel(
            model_name=args.model,
            species=args.species,
        )
        options["model"] = model.name
        options["media"] = "NoMedia"

        def runner(adata, **compass_options):
            return scCompass.compass(
                adata,
                model_loader=lambda name, species: model,
                **compass_options,
            )

    artifacts = run_pipeline(
        args.input,
        args.output_dir,
        compass_runner=runner,
        base_compass_options=options,
        sample_size=args.sample_size,
        overwrite=args.overwrite,
        export_full_table=args.export_full_table,
        plot_top_reactions=args.plot_top_reactions,
    )
    print("scCompass pipeline complete")
    print(f"model_scope: {args.model_scope}")
    print(f"input: {Path(args.input).resolve()}")
    for name, path in artifacts.items():
        if path.exists():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
