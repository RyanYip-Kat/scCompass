"""Plotting helpers backed by Scanpy, Seaborn, and Matplotlib."""

import numpy as np
import pandas as pd
import warnings


def umap(result, color, basis="X_umap", show=False, **kwargs):
    """Plot result reactions or cell annotations on a preserved embedding."""

    if basis not in result.obsm:
        raise KeyError(f"result.obsm does not contain embedding {basis!r}")
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scCompass.pl requires scanpy; install scCompass[plot]"
        ) from exc
    colors = [color] if isinstance(color, str) else list(color)
    reaction_columns = list(
        result.uns.get("scCompass_reaction_columns", [])
    )
    reaction_colors = [name for name in colors if name in reaction_columns]

    plot_adata = result
    if reaction_colors:
        import anndata as ad
        import numpy as np

        plot_adata = ad.AnnData(
            X=np.zeros((result.n_obs, 1), dtype=float),
            obs=result.obs.copy(),
            obsm={basis: result.obsm[basis]},
        )
        scores = _score_frame(result, "reactions")
        for name in reaction_colors:
            position = reaction_columns.index(name)
            plot_adata.obs[name] = scores.iloc[:, position].to_numpy()

    return sc.pl.embedding(
        plot_adata,
        basis=basis,
        color=colors,
        show=show,
        return_fig=True,
        **kwargs,
    )


def _score_frame(result, key):
    from .io import to_dataframe

    return to_dataframe(result, key=key)


def _groups(result, groupby):
    if groupby not in result.obs:
        raise KeyError(f"AnnData.obs does not contain group column {groupby!r}")
    groups = result.obs[groupby].astype(str)
    if isinstance(result.obs[groupby].dtype, pd.CategoricalDtype):
        order = [
            str(value)
            for value in result.obs[groupby].cat.categories
            if str(value) in set(groups)
        ]
    else:
        order = [str(value) for value in pd.unique(groups)]
    return groups, order


def heatmap(
    result,
    groupby,
    features=None,
    key="reactions",
    top_n=30,
    zscore=True,
    cmap=None,
    figsize=None,
    show=True,
):
    """Plot a group-level reaction or metabolite score heatmap."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    from .reporting import differential, group_summary

    if features is None:
        features = differential(result, groupby=groupby, key=key).head(top_n).index
    features = [features] if isinstance(features, str) else list(features)
    summary = group_summary(result, groupby=groupby, key=key, features=features)
    groups, order = _groups(result, groupby)
    matrix = summary.pivot(index="group", columns="feature", values="mean").reindex(
        order
    )
    if zscore:
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=1).replace(0, 1)
        matrix = (matrix - mean) / scale
        cmap = cmap or "RdBu_r"
        center = 0
        label = "Group mean z-score"
    else:
        cmap = cmap or "viridis"
        center = None
        label = "Penalty score"
    if figsize is None:
        figsize = (
            min(14.0, max(5.0, 0.36 * len(matrix.columns))),
            max(3.0, 0.3 * len(matrix)),
        )
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=PendingDeprecationWarning, module="seaborn.matrix"
        )
        sns.heatmap(
            matrix,
            cmap=cmap,
            center=center,
            ax=axis,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"label": label},
        )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Compass metabolic score by group")
    axis.tick_params(axis="x", rotation=60, labelsize=7)
    axis.tick_params(axis="y", rotation=0, labelsize=7)
    if show:
        plt.show()
    return figure


def raincloud(
    result,
    groupby,
    features,
    key="reactions",
    max_points_per_group=100,
    random_state=0,
    figsize=None,
    show=True,
):
    """Plot selected feature distributions with violin/box or raw points."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [features] if isinstance(features, str) else list(features)
    if not features:
        raise ValueError("At least one feature is required")
    scores = _score_frame(result, key).loc[:, features]
    groups, order = _groups(result, groupby)
    rng = np.random.default_rng(random_state)
    palette = dict(zip(order, sns.color_palette("colorblind", len(order))))

    rows = []
    for feature in features:
        for group in order:
            values = scores.loc[groups == group, feature].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) > max_points_per_group:
                values = rng.choice(
                    values, size=max_points_per_group, replace=False
                )
            rows.extend(
                {"feature": feature, "group": group, "score": float(value)}
                for value in values
            )
    plot_data = pd.DataFrame(rows)
    ncols = min(2, len(features))
    nrows = int(np.ceil(len(features) / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 3.2 * nrows)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()
    for index, feature in enumerate(features):
        axis = axes[index]
        data = plot_data[plot_data["feature"] == feature]
        counts = data.groupby("group", observed=True).size().reindex(order)
        if counts.min() >= 10:
            sns.violinplot(
                data=data,
                x="group",
                y="score",
                hue="group",
                order=order,
                palette=palette,
                legend=False,
                inner=None,
                cut=0,
                linewidth=0.6,
                ax=axis,
            )
            sns.boxplot(
                data=data,
                x="group",
                y="score",
                order=order,
                color="white",
                width=0.16,
                showfliers=False,
                linewidth=0.7,
                ax=axis,
            )
        sns.stripplot(
            data=data,
            x="group",
            y="score",
            order=order,
            color="#222222",
            size=2,
            alpha=0.55,
            jitter=0.22,
            linewidth=0,
            ax=axis,
        )
        axis.set_title(feature, fontsize=8)
        axis.set_xlabel("")
        axis.set_ylabel("Penalty score")
        axis.tick_params(axis="x", rotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(False)
    for axis in axes[len(features):]:
        axis.set_visible(False)
    if show:
        plt.show()
    return figure


def metabolite_dotplot(
    result,
    groupby,
    features=None,
    top_n=20,
    direction="both",
    figsize=None,
    show=True,
):
    """Plot uptake and/or secretion penalties by group and metabolite."""

    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import seaborn as sns

    from .reporting import group_summary

    directions = (
        ["uptake", "secretion"] if direction == "both" else [direction]
    )
    summaries = {
        name: group_summary(result, groupby=groupby, key=name)
        for name in directions
    }
    if features is None:
        scores = []
        for summary in summaries.values():
            matrix = summary.pivot(
                index="feature", columns="group", values="mean"
            )
            scores.append(matrix.var(axis=1))
        ranking = pd.concat(scores, axis=1).mean(axis=1).sort_values(ascending=False)
        features = ranking.head(top_n).index.tolist()
    features = [features] if isinstance(features, str) else list(features)
    groups, order = _groups(result, groupby)
    palette = sns.color_palette("viridis", 256)
    cmap = colors.ListedColormap(palette)
    if figsize is None:
        figsize = (4.0 * len(directions), max(3.5, 0.25 * len(features)))
    figure, axes = plt.subplots(
        1,
        len(directions),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()
    for axis, name in zip(axes, directions):
        summary = summaries[name]
        summary = summary[summary["feature"].isin(features)]
        matrix = summary.pivot(
            index="feature", columns="group", values="mean"
        ).reindex(index=features, columns=order)
        fraction = summary.pivot(
            index="feature", columns="group", values="fraction_positive"
        ).reindex(index=features, columns=order)
        values = matrix.to_numpy(dtype=float)
        fraction_values = fraction.to_numpy(dtype=float)
        x, y = np.meshgrid(np.arange(len(order)), np.arange(len(features)))
        norm = colors.Normalize(
            vmin=np.nanmin(values), vmax=np.nanmax(values)
        )
        scatter = axis.scatter(
            x.ravel(),
            y.ravel(),
            c=values.ravel(),
            s=15 + 250 * fraction_values.ravel(),
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.25,
        )
        axis.set_xticks(np.arange(len(order)), order, rotation=60)
        axis.set_yticks(np.arange(len(features)), features)
        axis.set_title(f"{name.capitalize()} penalty")
        axis.set_xlabel("")
        axis.set_ylabel("Metabolite" if name == directions[0] else "")
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(False)
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Mean penalty score")
    if show:
        plt.show()
    return figure


def metabolite_volcano(
    result,
    groupby,
    group_a,
    group_b,
    direction="both",
    pathways=None,
    max_pathways=4,
    effect_threshold=0.2,
    fdr_threshold=0.05,
    max_labels=6,
    y_cap=20.0,
    figsize=None,
    show=True,
):
    """Volcano-like metabolite panels split by pathway and exchange direction."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    from .reporting import metabolite_differential

    table = metabolite_differential(
        result,
        groupby=groupby,
        group_a=group_a,
        group_b=group_b,
        direction=direction,
    )
    if table.empty:
        raise ValueError("No metabolite pathway values are available")
    if pathways is None:
        ranking = (
            table.groupby("pathway")["metabolite"]
            .nunique()
            .sort_values(ascending=False)
        )
        pathways = ranking.head(max_pathways).index.tolist()
    pathways = [pathways] if isinstance(pathways, str) else list(pathways)
    directions = (
        ["uptake", "secretion"] if direction == "both" else [direction]
    )
    table = table[
        table["pathway"].isin(pathways)
        & table["direction"].isin(directions)
    ]
    if table.empty:
        raise ValueError("No data remain after pathway filtering")
    table["neglog10_fdr"] = np.minimum(
        -np.log10(
            np.clip(table["pvalue_adj"].to_numpy(dtype=float), 1e-300, 1.0)
        ),
        y_cap,
    )
    palette = sns.color_palette("colorblind", len(pathways) * len(directions))
    if figsize is None:
        figsize = (5.2 * len(directions), 3.8 * len(pathways))
    figure, axes = plt.subplots(
        len(pathways),
        len(directions),
        figsize=figsize,
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )
    color_index = 0
    for row_index, pathway in enumerate(pathways):
        for column_index, direction_name in enumerate(directions):
            axis = axes[row_index, column_index]
            panel = table[
                (table["pathway"] == pathway)
                & (table["direction"] == direction_name)
            ].copy()
            color = palette[color_index]
            color_index += 1
            if panel.empty:
                axis.set_visible(False)
                continue
            x = panel["cohens_d"].to_numpy(dtype=float)
            y = panel["neglog10_fdr"].to_numpy(dtype=float)
            axis.scatter(
                x,
                y,
                s=18,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
            axis.axvline(0, color="#555555", linewidth=0.7, linestyle="--")
            axis.axvline(
                effect_threshold,
                color="#777777",
                linewidth=0.7,
                linestyle="--",
            )
            axis.axvline(
                -effect_threshold,
                color="#777777",
                linewidth=0.7,
                linestyle="--",
            )
            axis.axhline(
                -np.log10(fdr_threshold),
                color="#777777",
                linewidth=0.7,
                linestyle="--",
            )
            x_limit = max(1.0, float(np.nanmax(np.abs(x))) * 1.2)
            axis.set_xlim(-x_limit, x_limit)
            y_limit = max(-np.log10(fdr_threshold) * 1.2, float(y_cap) * 1.08)
            axis.set_ylim(0, y_limit)
            title = (
                f"{pathway} · {direction_name}"
                if len(pathways) > 1
                else direction_name.capitalize()
            )
            axis.set_title(
                title,
                fontsize=8,
                color="white",
                bbox={
                    "facecolor": color,
                    "edgecolor": "none",
                    "pad": 4.5,
                },
            )
            axis.tick_params(labelsize=7)
            axis.grid(False)

            significant = panel[
                (panel["pvalue_adj"] <= fdr_threshold)
                & (panel["cohens_d"].abs() >= effect_threshold)
            ].copy()
            if significant.empty:
                significant = panel.reindex(
                    (
                        panel["cohens_d"].abs()
                        * panel["neglog10_fdr"]
                    ).sort_values(ascending=False).index
                ).head(min(2, max_labels))
            else:
                significant["rank"] = (
                    significant["cohens_d"].abs()
                    * significant["neglog10_fdr"]
                )
                significant = significant.nlargest(
                    min(max_labels, len(significant)), "rank"
                )
            significant = significant.sort_values(
                "neglog10_fdr", ascending=False
            )
            left_labels = significant[significant["cohens_d"] < 0]
            right_labels = significant[significant["cohens_d"] >= 0]
            for side, side_labels in ((-1, left_labels), (1, right_labels)):
                if side_labels.empty:
                    continue
                y_positions = np.linspace(
                    y_limit * 0.22,
                    y_limit * 0.86,
                    len(side_labels),
                )
                for label_index, ((_, point), label_y) in enumerate(
                    zip(side_labels.iterrows(), y_positions)
                ):
                    x_text = side * x_limit * (0.38 + 0.08 * (label_index % 2))
                    axis.annotate(
                        str(point["metabolite"]),
                        xy=(point["cohens_d"], point["neglog10_fdr"]),
                        xytext=(x_text, label_y),
                        fontsize=5.8,
                        ha="left" if side > 0 else "right",
                        va="center",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.72,
                            "pad": 0.5,
                        },
                        arrowprops={
                            "arrowstyle": "-",
                            "color": "#444444",
                            "linewidth": 0.5,
                            "shrinkA": 1,
                            "shrinkB": 2,
                        },
                    )
            if row_index == len(pathways) - 1:
                axis.set_xlabel(
                    f"Cohen's d ({group_a} - {group_b})", fontsize=8
                )
            if column_index == 0:
                axis.set_ylabel(f"-log10(FDR), capped at {y_cap:g}", fontsize=8)
    figure.suptitle(
        f"Metabolite exchange score differences: {group_a} vs {group_b}",
        fontsize=9,
    )
    if show:
        plt.show()
    return figure


def metabolite_pathway_strip(
    result,
    groupby,
    group_a,
    group_b,
    pathways=None,
    max_pathways=40,
    fdr_threshold=0.1,
    label_top_n=8,
    figsize=None,
    show=True,
):
    """Plot metabolite-level pathway effects as a signed strip plot."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .reporting import metabolite_pathway_differential

    table = metabolite_pathway_differential(
        result,
        groupby=groupby,
        group_a=group_a,
        group_b=group_b,
    )
    if table.empty:
        raise ValueError("No metabolite-pathway differences are available")
    if pathways is None:
        ranking = (
            table.assign(abs_effect=table["cohens_d"].abs())
            .groupby("pathway")["abs_effect"]
            .max()
            .sort_values(ascending=False)
        )
        pathways = ranking.head(max_pathways).index.tolist()
    pathways = [pathways] if isinstance(pathways, str) else list(pathways)
    table = table[table["pathway"].isin(pathways)].copy()
    if table.empty:
        raise ValueError("No data remain after pathway filtering")
    table["abs_effect"] = table["cohens_d"].abs()
    table["neglog10_fdr"] = -np.log10(
        np.clip(table["pvalue_adj"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    table["significant"] = table["pvalue_adj"] <= fdr_threshold

    if figsize is None:
        figsize = (10.0, max(3.2, 0.34 * len(pathways)))
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=False)
    figure.subplots_adjust(left=0.24, right=0.98, top=0.88, bottom=0.32)
    rng = np.random.default_rng(1337)
    path_positions = {pathway: index for index, pathway in enumerate(pathways)}
    y = np.asarray(
        [path_positions[pathway] for pathway in table["pathway"]],
        dtype=float,
    )
    jitter = rng.uniform(-0.16, 0.16, size=len(table))
    sizes = np.where(table["significant"], 28, 14)
    alphas = np.where(table["significant"], 0.95, 0.35)
    max_effect = max(1.0, float(np.nanmax(table["abs_effect"])))
    for index, (_, point) in enumerate(table.iterrows()):
        color = "#C51B3A" if point["cohens_d"] > 0 else "#2C7BB6"
        axis.scatter(
            point["cohens_d"],
            y[index] + jitter[index],
            s=sizes[index],
            color=color,
            alpha=alphas[index],
            edgecolor="none",
            zorder=3,
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=0.9)
    axis.set_yticks(np.arange(len(pathways)), pathways)
    axis.set_ylim(len(pathways) - 0.5, -0.5)
    axis.set_xlim(-max_effect * 1.15, max_effect * 1.15)
    axis.set_xlabel(f"Cohen's d ({group_a} - {group_b})", fontsize=8)
    axis.set_ylabel("")
    axis.set_title(
        "Up- and Down-Regulated Metabolites Across Pathway Boundaries",
        fontsize=10,
    )
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)

    label_candidates = table[table["significant"]].copy()
    if label_candidates.empty:
        label_candidates = table.copy()
    label_candidates["rank"] = (
        label_candidates["abs_effect"] * label_candidates["neglog10_fdr"]
    )
    label_candidates = label_candidates.nlargest(
        min(label_top_n, len(label_candidates)), "rank"
    )
    for pathway, pathway_labels in label_candidates.groupby("pathway", sort=False):
        pathway_labels = pathway_labels.sort_values("rank", ascending=False)
        offsets = np.linspace(-0.3, 0.3, len(pathway_labels))
        for label_index, ((_, point), y_offset) in enumerate(
            zip(pathway_labels.iterrows(), offsets)
        ):
            base_y = path_positions[pathway]
            side = 1 if point["cohens_d"] >= 0 else -1
            x_offset = point["cohens_d"] + side * max_effect * (
                0.035 + 0.015 * (label_index % 2)
            )
            axis.annotate(
                str(point["metabolite"]),
                xy=(point["cohens_d"], base_y),
                xytext=(x_offset, base_y + y_offset),
                fontsize=5.5,
                ha="left" if side > 0 else "right",
                va="center",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.35,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#555555",
                    "linewidth": 0.4,
                    "shrinkA": 1,
                    "shrinkB": 2,
                },
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#C51B3A",
            markersize=5,
            label=f"Higher in {group_a}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2C7BB6",
            markersize=5,
            label=f"Higher in {group_b}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            alpha=0.35,
            markersize=4,
            label="Not significant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#222222",
            markersize=5,
            label=f"BH-adjusted p < {fdr_threshold:g}",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=4,
    )
    if show:
        plt.show()
    return figure


def network(
    result,
    features=None,
    groupby=None,
    group=None,
    metabolite=None,
    top_n=30,
    figsize=(8.0, 6.0),
    show=True,
):
    """Plot a compact bipartite reaction-metabolite network."""

    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import networkx as nx

    metadata = result.uns.get("scCompass_reaction_var")
    if metadata is None:
        raise ValueError("Result does not contain reaction metadata")
    for column in ("reactants", "products"):
        if column not in metadata:
            raise ValueError(
                "Reaction metadata must contain reactants and products columns"
            )
    if metabolite is not None:
        mask = metadata.apply(
            lambda row: metabolite
            in (str(row["reactants"]) + ";" + str(row["products"])).split(";"),
            axis=1,
        )
        features = metadata.index[mask].tolist()
    elif features is None:
        from .reporting import differential

        selected = []
        if groupby is not None:
            selected = differential(
                result, groupby=groupby
            ).head(top_n).index.tolist()
        if not selected:
            scores = result.obsm["scCompass_reactions"]
            variance = np.var(scores, axis=0)
            selected = (
                pd.Series(variance, index=result.uns["scCompass_reaction_columns"])
                .nlargest(top_n)
                .index.tolist()
            )
        features = selected
    features = [features] if isinstance(features, str) else list(features)
    features = features[:top_n]
    metadata = metadata.reindex(features)
    graph = nx.DiGraph()
    for reaction_id, row in metadata.iterrows():
        graph.add_node(reaction_id, kind="reaction")
        for metabolite_id in str(row["reactants"]).split(";"):
            if metabolite_id:
                graph.add_node(metabolite_id, kind="metabolite")
                graph.add_edge(metabolite_id, reaction_id, edge_type="reactant")
        for metabolite_id in str(row["products"]).split(";"):
            if metabolite_id:
                graph.add_node(metabolite_id, kind="metabolite")
                graph.add_edge(reaction_id, metabolite_id, edge_type="product")
    if graph.number_of_nodes() == 0:
        raise ValueError("No nodes remain after network filtering")

    reaction_columns = list(result.uns["scCompass_reaction_columns"])
    positions = {name: reaction_columns.index(name) for name in features}
    scores = result.obsm["scCompass_reactions"]
    if group is not None and groupby is not None:
        groups = result.obs[groupby].astype(str)
        values = scores[groups.to_numpy() == str(group)].mean(axis=0)
    else:
        values = scores.mean(axis=0)
    reaction_values = {
        reaction: float(values[positions[reaction]]) for reaction in features
    }
    layout = nx.spring_layout(graph, seed=1337, k=0.7)
    reaction_nodes = [
        node for node, attrs in graph.nodes(data=True) if attrs["kind"] == "reaction"
    ]
    metabolite_nodes = [
        node for node, attrs in graph.nodes(data=True) if attrs["kind"] == "metabolite"
    ]
    norm = colors.Normalize(
        vmin=min(reaction_values.values()),
        vmax=max(reaction_values.values()),
    )
    cmap = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    nx.draw_networkx_edges(
        graph,
        layout,
        ax=axis,
        arrows=True,
        arrowsize=8,
        width=0.7,
        alpha=0.45,
    )
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=metabolite_nodes,
        node_shape="o",
        node_size=90,
        node_color="#BDBDBD",
        edgecolors="black",
        linewidths=0.4,
        ax=axis,
    )
    reaction_collection = nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=reaction_nodes,
        node_shape="s",
        node_size=260,
        node_color=[reaction_values[node] for node in reaction_nodes],
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        edgecolors="black",
        linewidths=0.5,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        font_size=5.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.2},
        ax=axis,
    )
    axis.axis("off")
    colorbar = figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axis,
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label(
        f"Mean penalty score ({group})" if group is not None else "Mean penalty score"
    )
    if show:
        plt.show()
    return figure
