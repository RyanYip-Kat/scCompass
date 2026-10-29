"""Command-line entry point for the full scCompass analysis pipeline."""

import argparse
import json

from .pipeline import run_pipeline


def _parser():
    parser = argparse.ArgumentParser(
        prog="sccompass-pipeline",
        description="Run the complete scCompass analysis on an AnnData file.",
    )
    parser.add_argument(
        "--input",
        default="testdata/pbmc5k_querydata.h5ad",
    )
    parser.add_argument("--output-dir", default="outputs/pbmc5k_pipeline")
    parser.add_argument("--model", default="RECON2_mat")
    parser.add_argument("--media", default="default-media")
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument("--layer", default="counts")
    parser.add_argument("--gene-symbol-key", default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--checkpoint-batch-size", type=int, default=64)
    parser.add_argument("--microcluster-size", type=int, default=None)
    parser.add_argument(
        "--solver-backend",
        choices=["cplex", "highs"],
        default="cplex",
    )
    parser.add_argument(
        "--penalty-backend",
        choices=["vectorized", "legacy"],
        default="vectorized",
    )
    parser.add_argument("--cell-chunk-size", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--plot-top-reactions", type=int, default=4)
    parser.add_argument("--export-full-table", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
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
        "gene_symbol_key": args.gene_symbol_key,
        "lambda_": args.lambda_,
        "num_neighbors": args.num_neighbors,
        "num_threads": args.num_threads,
        "num_processes": args.num_processes,
        "checkpoint_batch_size": args.checkpoint_batch_size,
        "microcluster_size": args.microcluster_size,
        "solver_backend": args.solver_backend,
        "penalty_backend": args.penalty_backend,
        "cell_chunk_size": args.cell_chunk_size,
        "test_mode": args.test_mode,
        "overwrite_checkpoint": args.overwrite_checkpoint,
    }
    artifacts = run_pipeline(
        args.input,
        args.output_dir,
        base_compass_options=options,
        sample_size=args.sample_size,
        overwrite=args.overwrite,
        export_full_table=args.export_full_table,
        plot_top_reactions=args.plot_top_reactions,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
