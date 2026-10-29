from scCompass.submodels import build_central_carbon_submodel


def test_central_carbon_submodel_is_compact_and_contains_named_reactions():
    model = build_central_carbon_submodel()

    assert model.name == "RECON2_CENTRAL_CARBON"
    assert len(model.reactions) < 1_000
    assert "ACONT_pos" in model.reactions
    assert "CSm_pos" in model.reactions
