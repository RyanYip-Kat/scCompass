"""Command-line entry point for scCompass."""

import argparse


def _parser():
    parser = argparse.ArgumentParser(
        prog="sccompass",
        description="Run Compass on an AnnData single-cell expression matrix.",
    )
    parser.add_argument("--input", required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output result .h5ad file")
    parser.add_argument("--model", default="RECON2_mat")
    parser.add_argument("--solver-backend", choices=["cplex", "highs"], default="cplex")
    parser.add_argument("--media", default="default-media")
    parser.add_argument("--species", default="homo_sapiens")
    parser.add_argument(
        "--layer",
        default="counts",
        help="Expression layer, or X to use AnnData.X",
    )
    parser.add_argument("--gene-symbol-key", default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--microcluster-size", type=int, default=None)
    parser.add_argument("--microcluster-filter", default="fano")
    parser.add_argument("--microcluster-filter-threshold", default=None, type=int)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-batch-size", type=int, default=64)
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--no-metabolites", action="store_true")
    parser.add_argument("--table", default=None, help="Optional reaction TSV output")
    return parser


def main(argv=None):
    from anndata import read_h5ad

    import scCompass

    args = _parser().parse_args(argv)
    adata = read_h5ad(args.input, backed="r" if args.sample_size else None)
    result = scCompass.compass(
        adata,
        model=args.model,
        solver_backend=args.solver_backend,
        media=args.media,
        species=args.species,
        expression_layer=None if args.layer == "X" else args.layer,
        gene_symbol_key=args.gene_symbol_key,
        lambda_=args.lambda_,
        num_neighbors=args.num_neighbors,
        calc_metabolites=not args.no_metabolites,
        test_mode=args.test_mode,
        num_threads=args.num_threads,
        num_processes=args.num_processes,
        sample_size=args.sample_size,
        microcluster_size=args.microcluster_size,
        microcluster_filter=args.microcluster_filter,
        microcluster_filter_threshold=args.microcluster_filter_threshold,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_batch_size=args.checkpoint_batch_size,
        overwrite_checkpoint=args.overwrite_checkpoint,
    )
    scCompass.io.save_h5ad(result, args.output)
    if args.table:
        scCompass.io.export_table(result, args.table)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
