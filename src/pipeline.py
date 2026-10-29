"""Reproducible end-to-end pipeline for AnnData Compass analysis."""

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ._version import __version__

RESULT_SCHEMA_VERSION = 2


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact_paths(output_dir):
    return {
        "result": output_dir / "result.h5ad",
        "score_result": output_dir / "scores.h5ad",
        "manifest": output_dir / "manifest.json",
        "reaction_summary": output_dir / "reaction_summary.tsv",
        "cell_summary": output_dir / "cell_summary.tsv",
        "plot": output_dir / "embedding.png",
        "full_table": output_dir / "reaction_scores.tsv.gz",
    }


def _normalized_options(options):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in options.items()
    }


def _config_signature(input_path, pipeline_config):
    return json.dumps(
        {
            "input": str(input_path),
            "pipeline": pipeline_config,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _write_full_table(result, path, chunk_size=512):
    scores = result.obsm["scCompass_reactions"]
    columns = result.uns["scCompass_reaction_columns"]
    with gzip.open(path, "wt", compresslevel=6) as handle:
        handle.write("\t" + "\t".join(map(str, columns)) + "\n")
        for start in range(0, result.n_obs, chunk_size):
            stop = min(result.n_obs, start + chunk_size)
            frame = pd.DataFrame(
                np.asarray(scores[start:stop]),
                index=result.obs_names[start:stop],
                columns=columns,
            )
            frame.to_csv(handle, sep="\t", header=False)


def _summaries(result, chunk_size=512):
    scores = np.asarray(result.obsm["scCompass_reactions"])
    reaction_columns = result.uns["scCompass_reaction_columns"]
    n_obs, n_features = scores.shape
    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_sumsq = np.zeros(n_features, dtype=np.float64)
    feature_min = np.full(n_features, np.inf)
    feature_max = np.full(n_features, -np.inf)
    feature_positive = np.zeros(n_features, dtype=np.int64)
    cell_sum = np.zeros(n_obs, dtype=np.float64)
    cell_sumsq = np.zeros(n_obs, dtype=np.float64)
    cell_positive = np.zeros(n_obs, dtype=np.int64)

    for start in range(0, n_obs, chunk_size):
        stop = min(n_obs, start + chunk_size)
        values = np.asarray(scores[start:stop], dtype=np.float64)
        feature_sum += values.sum(axis=0)
        feature_sumsq += np.square(values).sum(axis=0)
        feature_min = np.minimum(feature_min, values.min(axis=0))
        feature_max = np.maximum(feature_max, values.max(axis=0))
        feature_positive += (values > 0).sum(axis=0)
        cell_sum[start:stop] = values.sum(axis=1)
        cell_sumsq[start:stop] = np.square(values).sum(axis=1)
        cell_positive[start:stop] = (values > 0).sum(axis=1)

    denominator = max(n_obs, 1)
    feature_mean = feature_sum / denominator
    feature_var = np.maximum(feature_sumsq / denominator - np.square(feature_mean), 0)
    reaction_summary = pd.DataFrame(
        {
            "mean": feature_mean,
            "std": np.sqrt(feature_var),
            "min": feature_min,
            "max": feature_max,
            "fraction_positive": feature_positive / denominator,
        },
        index=reaction_columns.copy(),
    ).sort_index()

    cell_mean = cell_sum / max(n_features, 1)
    cell_var = np.maximum(
        cell_sumsq / max(n_features, 1) - np.square(cell_mean), 0
    )
    cell_summary = pd.DataFrame(
        {
            "mean": cell_mean,
            "std": np.sqrt(cell_var),
            "fraction_positive": cell_positive / max(n_features, 1),
        },
        index=result.obs_names.copy(),
    )
    top_reactions = reaction_summary["std"].nlargest(min(10, n_features)).index
    return reaction_summary, cell_summary, top_reactions


def _write_plot(result, path, top_reactions, plot_top_reactions):
    if "X_umap" not in result.obsm:
        return None
    import matplotlib.pyplot as plt

    from . import pl

    colors = list(top_reactions[:plot_top_reactions])
    if "celltype" in result.obs:
        colors.insert(0, "celltype")
    figure = pl.umap(
        result,
        color=colors,
        ncols=min(2, len(colors)),
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run_pipeline(
    input_path,
    output_dir,
    base_compass_options=None,
    sample_size=None,
    overwrite=False,
    export_full_table=False,
    plot_top_reactions=4,
    compass_runner=None,
):
    """Run the complete AnnData Compass analysis and write standard artifacts."""

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_paths(output_dir)
    options = dict(base_compass_options or {})
    options.setdefault("expression_layer", "counts")
    options.setdefault("checkpoint_dir", output_dir / "checkpoint")
    options.setdefault("overwrite_checkpoint", False)
    if sample_size is not None:
        options["sample_size"] = sample_size
    comparable_options = {
        key: value
        for key, value in _normalized_options(options).items()
        if key != "overwrite_checkpoint"
    }
    pipeline_config = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "sample_size": sample_size,
        "export_full_table": export_full_table,
        "plot_top_reactions": plot_top_reactions,
        "compass_options": comparable_options,
    }
    config_signature = _config_signature(input_path, pipeline_config)

    completed = (
        artifacts["manifest"].exists()
        and artifacts["result"].exists()
        and json.loads(artifacts["manifest"].read_text()).get("status") == "success"
    )
    if completed and not overwrite:
        manifest = json.loads(artifacts["manifest"].read_text())
        if manifest.get("config_signature") != config_signature:
            raise ValueError(
                "Existing output was produced with a different pipeline "
                "configuration; use overwrite=True or a new output directory"
            )
        return artifacts
    if artifacts["result"].exists() and not overwrite:
        raise FileExistsError(
            f"{artifacts['result']} exists without a completed manifest; "
            "use overwrite=True or a new output directory"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    from anndata import read_h5ad

    adata = read_h5ad(input_path, backed="r")

    if compass_runner is None:
        from . import compass as compass_runner

    result = compass_runner(adata, **options)
    reaction_summary, cell_summary, top_reactions = _summaries(result)
    reaction_summary.to_csv(artifacts["reaction_summary"], sep="\t")
    cell_summary.to_csv(artifacts["cell_summary"], sep="\t")

    if export_full_table:
        _write_full_table(result, artifacts["full_table"])
    elif artifacts["full_table"].exists() and overwrite:
        artifacts["full_table"].unlink()

    temporary_result = artifacts["result"].with_suffix(".h5ad.tmp")
    result.write_h5ad(temporary_result, compression="gzip")
    os.replace(temporary_result, artifacts["result"])
    from .result import to_score_anndata

    to_score_anndata(result).write_h5ad(
        artifacts["score_result"], compression="gzip"
    )
    _write_plot(
        result,
        artifacts["plot"],
        top_reactions,
        plot_top_reactions,
    )

    manifest = {
        "status": "success",
        "version": __version__,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
        },
        "config": {
            "sample_size": sample_size,
            "export_full_table": export_full_table,
            "plot_top_reactions": plot_top_reactions,
            "compass_options": _normalized_options(options),
        },
        "config_signature": config_signature,
        "result": {
            "n_obs": int(result.n_obs),
            "n_reactions": int(result.obsm["scCompass_reactions"].shape[1]),
            "input_n_vars": int(result.n_vars),
            "input_layers": list(result.layers),
            "has_input_raw": result.raw is not None,
            "obs_columns": list(result.obs.columns),
            "uns_scCompass": result.uns.get("scCompass", {}),
        },
        "outputs": {
            key: str(path)
            for key, path in artifacts.items()
            if path.exists()
        },
        "top_variable_reactions": list(map(str, top_reactions)),
    }
    _atomic_json(artifacts["manifest"], manifest)
    return artifacts
