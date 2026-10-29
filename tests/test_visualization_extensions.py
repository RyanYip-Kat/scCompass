import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib.figure import Figure

import scCompass.pl as scc_pl
from scCompass.reporting import (
    differential,
    group_summary,
    pathway_summary,
    run_report,
)
from scCompass.result import build_result, to_score_anndata


def _result():
    source = ad.AnnData(
        X=np.zeros((9, 2)),
        obs=pd.DataFrame(
            {
                "celltype": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                "donor": ["d1", "d2", "d1"] * 3,
            },
            index=[f"c{i}" for i in range(1, 10)],
        ),
    )
    source.obsm["X_umap"] = np.column_stack(
        [np.linspace(0, 1, 9), np.linspace(1, 0, 9)]
    )
    scores = pd.DataFrame(
        {
            "R1": [1.0, 1.1, 0.9, 2.0, 2.1, 1.9, 3.0, 3.1, 2.9],
            "R2": [3.0, 3.1, 2.9, 2.0, 2.1, 1.9, 1.0, 1.1, 0.9],
            "R3": [1.5] * 9,
            "EXC_TEST_pos": [2.0] * 9,
        },
        index=source.obs_names,
    )
    metadata = pd.DataFrame(
        {
            "subsystem": ["Glycolysis", "TCA", "TCA", ""],
            "reactants": ["M1", "M2", "M3", "M4"],
            "products": ["M2", "M3", "M4", ""],
            "reaction_name": ["R1", "R2", "R3", "EXC_TEST"],
            "is_exchange": [False, False, False, True],
        },
        index=["R1", "R2", "R3", "EXC_TEST_pos"],
    )
    uptake = pd.DataFrame(
        {
            "M1": np.linspace(0.1, 0.9, 9),
            "M2": np.linspace(0.9, 0.1, 9),
        },
        index=source.obs_names,
    )
    secretion = uptake.iloc[::-1].copy()
    secretion.index = source.obs_names
    return build_result(
        source,
        scores,
        reaction_metadata=metadata,
        uptake=uptake,
        secretion=secretion,
    )


def test_group_summary_and_differential_detect_reaction_changes():
    result = _result()

    summary = group_summary(result, groupby="celltype")
    diff = differential(result, groupby="celltype")

    r1 = summary[(summary["feature"] == "R1") & (summary["group"] == "C")]
    assert np.isclose(r1["mean"].iloc[0], 3.0)
    assert summary.loc[summary["feature"] == "R3", "std"].max() == 0
    assert diff.loc["R1", "best_group"] == "A"
    assert diff.loc["R1", "worst_group"] == "C"
    assert diff.loc["R1", "pvalue_adj"] < 0.1
    assert "EXC_TEST_pos" not in summary["feature"].unique()
    assert "EXC_TEST_pos" not in diff.index


def test_pathway_summary_aggregates_reaction_metadata():
    pathway = pathway_summary(_result(), groupby="celltype")

    row = pathway[
        (pathway["pathway"] == "TCA") & (pathway["group"] == "A")
    ].iloc[0]
    assert np.isclose(row["mean"], 2.25)
    assert row["n_reactions"] == 2


def test_reactome_heatmap_and_raincloud_return_figures():
    result = _result()

    heatmap = scc_pl.heatmap(
        result,
        groupby="celltype",
        features=["R1", "R2", "R3"],
    )
    raincloud = scc_pl.raincloud(
        result,
        groupby="celltype",
        features=["R1", "R2"],
    )

    assert isinstance(heatmap, Figure)
    assert isinstance(raincloud, Figure)
    assert len(raincloud.axes) >= 2


def test_metabolite_dotplot_and_network_return_figures():
    result = _result()

    dotplot = scc_pl.metabolite_dotplot(
        result,
        groupby="celltype",
        features=["M1", "M2"],
    )
    network = scc_pl.network(
        result,
        features=["R1", "R2"],
    )
    volcano = scc_pl.metabolite_volcano(
        result,
        groupby="celltype",
        group_a="A",
        group_b="B",
    )

    assert isinstance(dotplot, Figure)
    assert isinstance(network, Figure)
    assert isinstance(volcano, Figure)
    assert len(network.axes) >= 1


def test_report_writes_tables_and_figures(tmp_path):
    artifacts = run_report(
        _result(),
        tmp_path / "report",
        groupby="celltype",
        features=["R1", "R2"],
    )

    assert artifacts["group_summary"].exists()
    assert artifacts["differential"].exists()
    assert artifacts["pathway_summary"].exists()
    assert artifacts["heatmap"].exists()
    assert artifacts["umap"].exists()
    assert artifacts["raincloud"].exists()
    assert artifacts["metabolite_dotplot"].exists()
    assert artifacts["metabolite_differential"].exists()
    assert artifacts["metabolite_volcano"].exists()
    assert artifacts["network"].exists()


def test_umap_accepts_legacy_score_centric_anndata():
    result = to_score_anndata(_result())

    figure = scc_pl.umap(result, color=["R1", "celltype"])

    assert isinstance(figure, Figure)
