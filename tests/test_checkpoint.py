import json

import numpy as np
import pandas as pd
import pytest

from scCompass.checkpoint import (
    BatchCheckpoint,
    CellResult,
    compute_checkpoint_fingerprint,
)
from scCompass.exceptions import CheckpointMismatchError


def _cell(name, reactions, uptake=None):
    return CellResult(
        cell=name,
        reactions=pd.Series(reactions, dtype=float),
        uptake=None if uptake is None else pd.Series(uptake, dtype=float),
    )


def test_checkpoint_writes_atomic_npz_batches_and_collects_cells(tmp_path):
    store = BatchCheckpoint(tmp_path / "checkpoint", fingerprint="abc", batch_size=2)
    store.write_batch(
        [
            _cell("c1", {"R1": 1.0, "R2": 2.0}, {"M1": 3.0}),
            _cell("c2", {"R1": 4.0}, {"M2": 5.0}),
        ]
    )
    store.write_batch([_cell("c3", {"R2": 6.0})])

    assert store.completed_cells() == {"c1", "c2", "c3"}
    assert not list(store.directory.rglob("*.tsv"))
    assert not list(store.directory.rglob("*.tmp"))

    frames = store.load_frames(["c3", "c1", "c2"])
    assert frames["reactions"].index.tolist() == ["c3", "c1", "c2"]
    assert frames["reactions"].columns.tolist() == ["R1", "R2"]
    assert frames["reactions"].loc["c3", "R1"] != frames["reactions"].loc["c3", "R1"]
    assert frames["uptake"].columns.tolist() == ["M1", "M2"]
    assert frames["uptake"].loc["c1", "M2"] != frames["uptake"].loc["c1", "M2"]

    manifest = json.loads((store.directory / "manifest.json").read_text())
    assert manifest["fingerprint"] == "abc"
    assert manifest["schema_version"] == 1


def test_checkpoint_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "checkpoint"
    BatchCheckpoint(path, fingerprint="abc")

    with pytest.raises(CheckpointMismatchError, match="different input or parameters"):
        BatchCheckpoint(path, fingerprint="different")


def test_checkpoint_overwrite_removes_only_owned_batch_files(tmp_path):
    path = tmp_path / "checkpoint"
    store = BatchCheckpoint(path, fingerprint="abc")
    store.write_batch([_cell("c1", {"R1": 1.0})])
    sentinel = path / "user-file.txt"
    sentinel.write_text("keep")

    restarted = BatchCheckpoint(path, fingerprint="next", overwrite=True)

    assert restarted.completed_cells() == set()
    assert sentinel.read_text() == "keep"


def test_fingerprint_changes_when_expression_or_parameters_change():
    expression = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["G1", "G2"],
        columns=["c1", "c2"],
    )
    model = type("Model", (), {"name": "test-model", "reactions": {}})()
    args = {"lambda": 0.0, "calc_metabolites": True, "num_threads": 1}

    first = compute_checkpoint_fingerprint(expression, model, args)
    repeated = compute_checkpoint_fingerprint(expression.copy(), model, args.copy())
    changed_expression = expression.copy()
    changed_expression.iloc[0, 0] = 9.0
    changed_args = {**args, "lambda": 0.25}

    assert repeated == first
    assert compute_checkpoint_fingerprint(changed_expression, model, args) != first
    assert compute_checkpoint_fingerprint(expression, model, changed_args) != first


def test_fingerprint_ignores_execution_only_settings():
    expression = pd.DataFrame([[1.0]], index=["G1"], columns=["c1"])
    model = type("Model", (), {"name": "test-model", "reactions": {}})()

    first = compute_checkpoint_fingerprint(
        expression, model, {"lambda": 0.0, "num_threads": 1}
    )
    changed = compute_checkpoint_fingerprint(
        expression, model, {"lambda": 0.0, "num_threads": 8}
    )

    assert first == changed
