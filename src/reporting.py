"""Summary tables and reusable reporting functions for scCompass results."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def _score_frame(result, key):
    from .io import to_dataframe

    return to_dataframe(result, key=key)


def _group_series(result, groupby):
    if groupby not in result.obs:
        raise KeyError(f"AnnData.obs does not contain group column {groupby!r}")
    series = result.obs[groupby].astype(str)
    if series.isna().any():
        raise ValueError(f"Group column {groupby!r} contains missing values")
    return series


def _ordered_groups(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = series.cat.categories.tolist()
    else:
        values = pd.unique(series).tolist()
    present = set(series.tolist())
    return [str(value) for value in values if str(value) in present]


def group_summary(
    result,
    groupby,
    key="reactions",
    features=None,
    include_exchange=False,
):
    """Return long-form group summaries for reaction or metabolite scores."""

    scores = _score_frame(result, key)
    if features is not None:
        features = [features] if isinstance(features, str) else list(features)
        missing = [feature for feature in features if feature not in scores]
        if missing:
            raise KeyError(f"Unknown {key} features: {missing}")
        scores = scores.loc[:, features]
    elif key == "reactions" and not include_exchange:
        metadata = result.uns.get("scCompass_reaction_var")
        if metadata is not None and "is_exchange" in metadata:
            is_exchange = (
                metadata["is_exchange"]
                .astype(bool)
                .reindex(scores.columns)
                .fillna(False)
            )
            scores = scores.loc[:, ~is_exchange]
    groups = _group_series(result, groupby)
    rows = []
    for group in _ordered_groups(groups):
        values = scores.loc[groups == group]
        for feature in values.columns:
            data = values[feature].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "feature": str(feature),
                    "n": int(data.size),
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
                    "fraction_positive": float(np.mean(data > 0)),
                }
            )
    return pd.DataFrame(rows)


def _adjust_bh(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 1.0
    count = len(pvalues)
    for rank in range(count - 1, -1, -1):
        index = order[rank]
        running = min(running, pvalues[index] * count / (rank + 1))
        adjusted[index] = running
    return adjusted


def differential(
    result,
    groupby,
    key="reactions",
    features=None,
    include_exchange=False,
):
    """Calculate feature-level multi-group differences and effect size."""

    summary = group_summary(
        result,
        groupby,
        key=key,
        features=features,
        include_exchange=include_exchange,
    )
    mean_table = summary.pivot(index="feature", columns="group", values="mean")
    median_table = summary.pivot(index="feature", columns="group", values="median")
    scores = _score_frame(result, key).loc[:, mean_table.index]
    groups = _group_series(result, groupby)
    group_order = mean_table.columns.tolist()

    rows = []
    for feature in mean_table.index:
        arrays = [
            scores.loc[groups == group, feature].to_numpy(dtype=float)
            for group in group_order
        ]
        arrays = [array[np.isfinite(array)] for array in arrays]
        if len(arrays) == 2:
            statistic, pvalue = mannwhitneyu(
                arrays[0], arrays[1], alternative="two-sided"
            )
            effect = float(
                1
                - 2 * statistic / (len(arrays[0]) * len(arrays[1]))
            )
        else:
            pooled = np.concatenate(arrays)
            if np.unique(pooled).size <= 1:
                statistic, pvalue = 0.0, 1.0
            else:
                statistic, pvalue = kruskal(*arrays)
            sample_count = sum(len(array) for array in arrays)
            effect = float(
                max(
                    0.0,
                    (statistic - len(arrays) + 1)
                    / max(sample_count - len(arrays), 1),
                )
            )
        best = median_table.loc[feature].idxmin()
        worst = median_table.loc[feature].idxmax()
        row = {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "effect_size": effect,
            "best_group": str(best),
            "worst_group": str(worst),
            "worst_minus_best": float(
                mean_table.loc[feature, worst] - mean_table.loc[feature, best]
            ),
        }
        row.update(
            {
                f"mean:{group}": float(mean_table.loc[feature, group])
                for group in group_order
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows, index=mean_table.index)
    output.index.name = "feature"
    output["pvalue_adj"] = _adjust_bh(output["pvalue"])
    return output.sort_values(
        ["pvalue_adj", "effect_size"], ascending=[True, False]
    )


def pathway_summary(
    result,
    groupby,
    key="reactions",
    include_exchange=False,
):
    """Aggregate cell-level mean reaction penalties by annotated subsystem."""

    scores = _score_frame(result, key)
    metadata = result.uns.get("scCompass_reaction_var")
    if metadata is None or "subsystem" not in metadata:
        raise ValueError("Result does not contain reaction subsystem metadata")
    metadata = metadata.reindex(scores.columns)
    if not include_exchange and "is_exchange" in metadata:
        keep = ~metadata["is_exchange"].astype(bool).fillna(False)
        metadata = metadata.loc[keep]
        scores = scores.loc[:, metadata.index]
    mapping = metadata["subsystem"].reindex(scores.columns).fillna("Unassigned")
    groups = _group_series(result, groupby)
    rows = []
    for pathway in sorted(set(mapping.astype(str))):
        columns = mapping.index[mapping.astype(str) == pathway]
        pathway_scores = scores.loc[:, columns].mean(axis=1)
        for group in _ordered_groups(groups):
            values = pathway_scores.loc[groups == group].to_numpy(dtype=float)
            rows.append(
                {
                    "pathway": pathway,
                    "group": group,
                    "n": int(values.size),
                    "n_reactions": int(len(columns)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1))
                    if values.size > 1
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _metabolite_pathways(result):
    metadata = result.uns.get("scCompass_reaction_var")
    required = {"reactants", "products", "subsystem", "is_exchange"}
    if metadata is None or not required.issubset(metadata.columns):
        raise ValueError(
            "Result does not contain reaction network and subsystem metadata"
        )
    pairs = []
    for reaction_id, row in metadata.iterrows():
        if bool(row["is_exchange"]):
            continue
        pathway = str(row["subsystem"]).strip()
        if not pathway:
            continue
        metabolites = set(str(row["reactants"]).split(";"))
        metabolites.update(str(row["products"]).split(";"))
        for metabolite in metabolites:
            metabolite = metabolite.strip()
            if metabolite and metabolite.lower() != "nan":
                pairs.append(
                    {
                        "metabolite": metabolite,
                        "pathway": pathway,
                        "reaction": str(reaction_id),
                    }
                )
    return pd.DataFrame(pairs).drop_duplicates()


def metabolite_differential(
    result,
    groupby,
    group_a,
    group_b,
    direction="both",
):
    """Compare metabolite exchange penalties between two selected groups."""

    groups = _group_series(result, groupby)
    missing_groups = {str(group_a), str(group_b)}.difference(groups.unique())
    if missing_groups:
        raise ValueError(f"Unknown groups: {sorted(missing_groups)}")
    if str(group_a) == str(group_b):
        raise ValueError("group_a and group_b must be different")

    links = _metabolite_pathways(result)
    directions = (
        ["uptake", "secretion"] if direction == "both" else [direction]
    )
    if any(name not in {"uptake", "secretion"} for name in directions):
        raise ValueError("direction must be uptake, secretion, or both")

    rows = []
    for direction_name in directions:
        scores = _score_frame(result, direction_name)
        for pathway, pathway_links in links.groupby("pathway", sort=True):
            metabolites = [
                metabolite
                for metabolite in pathway_links["metabolite"].unique()
                if metabolite in scores.columns
            ]
            group_rows = []
            for metabolite in metabolites:
                values_a = scores.loc[
                    groups == str(group_a), metabolite
                ].to_numpy(dtype=float)
                values_b = scores.loc[
                    groups == str(group_b), metabolite
                ].to_numpy(dtype=float)
                values_a = values_a[np.isfinite(values_a)]
                values_b = values_b[np.isfinite(values_b)]
                if values_a.size < 2 or values_b.size < 2:
                    continue
                mean_a = float(np.mean(values_a))
                mean_b = float(np.mean(values_b))
                variance_a = float(np.var(values_a, ddof=1))
                variance_b = float(np.var(values_b, ddof=1))
                pooled = np.sqrt(
                    (
                        (values_a.size - 1) * variance_a
                        + (values_b.size - 1) * variance_b
                    )
                    / (values_a.size + values_b.size - 2)
                )
                if pooled == 0:
                    effect = 0.0 if np.isclose(mean_a, mean_b) else np.sign(
                        mean_a - mean_b
                    ) * 10.0
                else:
                    effect = float((mean_a - mean_b) / pooled)
                if np.all(values_a == values_a[0]) and np.all(
                    values_b == values_b[0]
                ):
                    pvalue = 1.0
                else:
                    _, pvalue = mannwhitneyu(
                        values_a, values_b, alternative="two-sided"
                    )
                group_rows.append(
                    {
                        "direction": direction_name,
                        "pathway": str(pathway),
                        "metabolite": str(metabolite),
                        "n_a": int(values_a.size),
                        "n_b": int(values_b.size),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "cohens_d": effect,
                        "pvalue": float(pvalue),
                    }
                )
            if group_rows:
                panel = pd.DataFrame(group_rows)
                panel["pvalue_adj"] = _adjust_bh(panel["pvalue"])
                rows.extend(panel.to_dict("records"))
    if not rows:
        return pd.DataFrame(
            columns=[
                "direction",
                "pathway",
                "metabolite",
                "n_a",
                "n_b",
                "mean_a",
                "mean_b",
                "cohens_d",
                "pvalue",
                "pvalue_adj",
            ]
        )
    return pd.DataFrame(rows)


def run_report(
    result,
    output_dir,
    groupby="celltype",
    features=None,
    top_n=30,
    include_exchange=False,
    volcano_groups=None,
    volcano_effect_threshold=0.2,
    volcano_fdr_threshold=0.05,
    volcano_max_labels=6,
    volcano_y_cap=20.0,
    output_format="png",
    dpi=300,
):
    """Generate standard Compass tables and figures from one result AnnData."""

    import matplotlib.pyplot as plt

    from . import pl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = {
        "group_summary": "reaction_group_summary.tsv",
        "differential": "reaction_differential.tsv",
        "pathway_summary": "pathway_summary.tsv",
        "metabolite_differential": "metabolite_differential.tsv",
        "umap": f"reaction_umap.{output_format}",
        "heatmap": f"reaction_heatmap.{output_format}",
        "raincloud": f"reaction_raincloud.{output_format}",
        "metabolite_dotplot": f"metabolite_dotplot.{output_format}",
        "network": f"metabolic_network.{output_format}",
        "metabolite_volcano": f"metabolite_volcano.{output_format}",
    }
    artifacts = {key: output_dir / value for key, value in suffixes.items()}

    summary = group_summary(
        result, groupby=groupby, include_exchange=include_exchange
    )
    diff = differential(
        result, groupby=groupby, include_exchange=include_exchange
    )
    pathways = pathway_summary(
        result, groupby=groupby, include_exchange=include_exchange
    )
    summary.to_csv(artifacts["group_summary"], sep="\t", index=False)
    diff.to_csv(artifacts["differential"], sep="\t")
    pathways.to_csv(artifacts["pathway_summary"], sep="\t", index=False)

    if volcano_groups is None:
        order = _ordered_groups(_group_series(result, groupby))
        preferred = ("CD4 Naive", "CD4 Tem")
        volcano_groups = (
            preferred
            if all(group in order for group in preferred)
            else tuple(order[:2])
        )
    if len(volcano_groups) != 2:
        raise ValueError("volcano_groups must contain exactly two groups")
    metabolite_diff = metabolite_differential(
        result,
        groupby=groupby,
        group_a=volcano_groups[0],
        group_b=volcano_groups[1],
    )
    metabolite_diff.to_csv(
        artifacts["metabolite_differential"], sep="\t", index=False
    )

    selected = (
        [features] if isinstance(features, str) else list(features or [])
    )
    if not selected:
        selected = diff.head(min(6, len(diff))).index.tolist()

    figures = [
        (
            "heatmap",
            pl.heatmap(
                result,
                groupby=groupby,
                features=diff.head(top_n).index,
            ),
        ),
        (
            "raincloud",
            pl.raincloud(
                result,
                groupby=groupby,
                features=selected,
            ),
        ),
        (
            "metabolite_dotplot",
            pl.metabolite_dotplot(result, groupby=groupby),
        ),
        (
            "network",
            pl.network(result, features=diff.head(top_n).index),
        ),
        (
            "metabolite_volcano",
            pl.metabolite_volcano(
                result,
                groupby=groupby,
                group_a=volcano_groups[0],
                group_b=volcano_groups[1],
                effect_threshold=volcano_effect_threshold,
                fdr_threshold=volcano_fdr_threshold,
                max_labels=volcano_max_labels,
                y_cap=volcano_y_cap,
            ),
        ),
    ]
    if "X_umap" in result.obsm:
        umap_colors = [groupby] + list(diff.head(4).index)
        figures.append(
            (
                "umap",
                pl.umap(
                    result,
                    color=umap_colors,
                    ncols=min(3, len(umap_colors)),
                ),
            )
        )
    for key, figure in figures:
        figure.savefig(artifacts[key], dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return artifacts
