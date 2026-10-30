#!/usr/bin/env python
"""Visualize and export results produced by scripts/pipeline.py."""

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import anndata as ad

import scCompass


def _parser():
    parser = argparse.ArgumentParser(
        description="Create Compass tables and figures from a saved result.h5ad."
    )
    parser.add_argument(
        "--input",
        default="outputs/pbmc5k_full_pipeline/result.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pbmc5k_visual_report",
    )
    parser.add_argument("--groupby", default="celltype")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated reactions for focused plots",
    )
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-exchange", action="store_true")
    parser.add_argument("--export-scores", action="store_true")
    parser.add_argument("--volcano-group-a", default=None)
    parser.add_argument("--volcano-group-b", default=None)
    parser.add_argument("--effect-threshold", type=float, default=0.2)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--volcano-max-labels", type=int, default=6)
    parser.add_argument("--volcano-y-cap", type=float, default=20.0)
    parser.add_argument("--pathway-fdr-threshold", type=float, default=0.1)
    parser.add_argument("--pathway-max-labels", type=int, default=8)
    return parser


def _export_scores(result, path):
    scores = scCompass.to_dataframe(result, key="reactions")
    with gzip.open(path, "wt", compresslevel=6) as handle:
        scores.to_csv(handle, sep="\t")
    return path


def main(argv=None):
    args = _parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = ad.read_h5ad(input_path, backed="r")
    features = None
    if args.features:
        features = [
            value.strip() for value in args.features.split(",") if value.strip()
        ]
    if bool(args.volcano_group_a) != bool(args.volcano_group_b):
        raise ValueError(
            "--volcano-group-a and --volcano-group-b must be supplied together"
        )
    volcano_groups = (
        (args.volcano_group_a, args.volcano_group_b)
        if args.volcano_group_a
        else None
    )

    artifacts = scCompass.run_report(
        result,
        output_dir,
        groupby=args.groupby,
        features=features,
        top_n=args.top_n,
        include_exchange=args.include_exchange,
        volcano_groups=volcano_groups,
        volcano_effect_threshold=args.effect_threshold,
        volcano_fdr_threshold=args.fdr_threshold,
        volcano_max_labels=args.volcano_max_labels,
        volcano_y_cap=args.volcano_y_cap,
        pathway_fdr_threshold=args.pathway_fdr_threshold,
        pathway_max_labels=args.pathway_max_labels,
        output_format=args.format,
        dpi=args.dpi,
    )
    if args.export_scores:
        artifacts["scores"] = _export_scores(
            result, output_dir / "reaction_scores.tsv.gz"
        )
    result.file.close()

    manifest = {
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "groupby": args.groupby,
        "top_n": args.top_n,
        "include_exchange": args.include_exchange,
        "features": features,
        "volcano_groups": volcano_groups,
        "effect_threshold": args.effect_threshold,
        "fdr_threshold": args.fdr_threshold,
        "volcano_y_cap": args.volcano_y_cap,
        "pathway_fdr_threshold": args.pathway_fdr_threshold,
        "pathway_max_labels": args.pathway_max_labels,
        "outputs": {
            key: str(path)
            for key, path in artifacts.items()
            if path.exists()
        },
    }
    manifest_path = output_dir / "visual_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    artifacts["manifest"] = manifest_path
    print(json.dumps({key: str(path) for key, path in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
