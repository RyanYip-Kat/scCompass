import inspect

import scCompass


def test_public_compass_api_is_available():
    assert callable(scCompass.compass)
    assert scCompass.compass is scCompass.tl.compass
    params = inspect.signature(scCompass.compass).parameters
    assert "adata" in params
    assert "expression_layer" in params
    assert "lambda_" in params


def test_result_and_io_helpers_are_exposed():
    assert callable(scCompass.to_dataframe)
    assert callable(scCompass.export_table)
    assert callable(scCompass.io.save_h5ad)
