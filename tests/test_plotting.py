import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib.figure import Figure

import scCompass.pl as scc_pl


def _result():
    source = ad.AnnData(
        X=np.zeros((3, 2)),
        obs=pd.DataFrame(
            {"celltype": ["T", "B", "T"]}, index=["c1", "c2", "c3"]
        ),
    )
    source.obsm["X_umap"] = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    from scCompass.result import build_result

    return build_result(
        source,
        pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            index=["c1", "c2", "c3"],
            columns=["R1", "R2"],
        ),
    )


def test_umap_uses_result_scores_without_recomputing_embedding():
    result = _result()
    figure = scc_pl.umap(result, color=["R1", "celltype"])

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 2
    assert not np.shares_memory(
        result.obsm["X_umap"], result.obsm["scCompass_reactions"]
    )
