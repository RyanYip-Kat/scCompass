import pandas as pd

from scCompass.compat import iteritems


def test_iteritems_matches_pandas_items_behavior():
    frame = pd.DataFrame({"a": [1], "b": [2]})

    items = list(iteritems(frame))
    assert [key for key, _ in items] == ["a", "b"]
    assert [value.tolist() for _, value in items] == [[1], [2]]


def test_mat_model_import_does_not_require_libsbml():
    from scCompass.core import models

    model = models.load_metabolic_model("RECON2_mat")

    assert model.name == "RECON2_mat"
    assert len(model.reactions) > 1_000
