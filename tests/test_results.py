import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scCompass.io import export_table, save_h5ad, to_dataframe
from scCompass.result import build_result, to_score_anndata


def _source():
    source = ad.AnnData(
        X=np.zeros((2, 3)),
        obs=pd.DataFrame({"celltype": ["T", "B"]}, index=["c1", "c2"]),
    )
    source.layers["counts"] = np.ones((2, 3))
    source.obsm["X_umap"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    source.obsp["connectivities"] = np.eye(2)
    source.varm["gene_meta"] = np.ones((3, 1))
    source.varp["gene_graph"] = np.eye(3)
    source.uns["experiment"] = {"name": "pbmc"}
    source.raw = source
    return source


def test_result_uses_cells_by_reactions_and_copies_visual_inputs():
    source = _source()
    scores = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=["c1", "c2"],
        columns=["R1", "R2"],
    )

    result = build_result(
        source,
        scores,
        reaction_metadata=pd.DataFrame(
            {"subsystem": ["Glycolysis", "TCA"]},
            index=["R1", "R2"],
        ),
        uptake=pd.DataFrame([[1.0], [2.0]], index=["c1", "c2"], columns=["M1"]),
        run_metadata={"model": "RECON2_mat"},
    )

    assert result.shape == source.shape
    assert result.obs_names.tolist() == ["c1", "c2"]
    assert result.var_names.tolist() == source.var_names.tolist()
    assert "counts" in result.layers
    assert "gene_meta" in result.varm
    assert "gene_graph" in result.varp
    assert result.raw is not None
    np.testing.assert_allclose(
        result.obsm["scCompass_reactions"], scores.to_numpy()
    )
    reaction_metadata = result.uns["scCompass_reaction_var"]
    assert reaction_metadata.loc["R2", "subsystem"] == "TCA"
    np.testing.assert_array_equal(result.obsm["X_umap"], source.obsm["X_umap"])
    np.testing.assert_array_equal(
        result.obsp["connectivities"], source.obsp["connectivities"]
    )
    assert result.uns["experiment"] == {"name": "pbmc"}
    assert result.uns["scCompass"]["model"] == "RECON2_mat"
    assert result.uns["scCompass"]["outputs"]["uptake"] == "scCompass_uptake"
    np.testing.assert_array_equal(
        result.obsm["scCompass_uptake"], np.array([[1.0], [2.0]])
    )


def test_result_accepts_reactions_by_cells_and_restores_cell_order():
    source = _source()
    legacy_scores = pd.DataFrame(
        [[0.2, 0.1], [0.4, 0.3]],
        index=["R1", "R2"],
        columns=["c2", "c1"],
    )

    result = build_result(source, legacy_scores)

    assert result.obs_names.tolist() == ["c1", "c2"]
    assert result.uns["scCompass_reaction_columns"].tolist() == ["R1", "R2"]
    np.testing.assert_allclose(
        result.obsm["scCompass_reactions"], [[0.1, 0.3], [0.2, 0.4]]
    )


def test_table_exports_match_result_views(tmp_path):
    result = build_result(
        _source(),
        pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=["c1", "c2"],
            columns=["R1", "R2"],
        ),
    )

    frame = to_dataframe(result)
    assert frame.index.tolist() == ["c1", "c2"]
    assert frame.columns.tolist() == ["R1", "R2"]

    path = tmp_path / "reactions.tsv"
    export_table(result, path)
    assert pd.read_csv(path, sep="\t", index_col=0).shape == (2, 2)


def test_save_h5ad_round_trips_result(tmp_path):
    result = build_result(
        _source(),
        pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=["c1", "c2"],
            columns=["R1", "R2"],
        ),
    )

    path = save_h5ad(result, tmp_path / "result.h5ad")
    restored = ad.read_h5ad(path)

    assert restored.shape == (2, 3)
    assert restored.obs_names.tolist() == ["c1", "c2"]


def test_score_view_preserves_legacy_cells_by_reactions_layout():
    result = build_result(
        _source(),
        pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=["c1", "c2"],
            columns=["R1", "R2"],
        ),
    )

    scores = to_score_anndata(result)

    assert scores.shape == (2, 2)
    assert scores.var_names.tolist() == ["R1", "R2"]
    np.testing.assert_allclose(scores.X, [[0.1, 0.2], [0.3, 0.4]])


def test_result_rejects_missing_cells():
    with pytest.raises(ValueError, match="same cells"):
        build_result(
            _source(),
            pd.DataFrame([[0.1, 0.2]], index=["other"], columns=["R1", "R2"]),
        )
