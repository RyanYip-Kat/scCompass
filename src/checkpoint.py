"""Atomic batch checkpoints for resumable Compass cell-level FBA results."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from .exceptions import CheckpointMismatchError, ScCompassError


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
FINGERPRINT_ARGUMENTS = (
    "lambda",
    "num_neighbors",
    "symmetric_kernel",
    "penalty_diffusion",
    "penalty_backend",
    "and_function",
    "isoform_summing",
    "select_reactions",
    "select_subsystems",
    "test_mode",
    "calc_metabolites",
    "lpmethod",
    "advance",
    "microcluster_size",
    "microcluster_filter",
    "microcluster_filter_threshold",
    "solver_backend",
)


@dataclass(frozen=True)
class CellResult:
    cell: str
    reactions: pd.Series
    uptake: pd.Series | None = None
    secretion: pd.Series | None = None


def _update_json_hash(digest, value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def compute_checkpoint_fingerprint(expression, model, args):
    """Hash the expression values, model structure, and solver semantics."""

    digest = hashlib.blake2b(digest_size=20)
    _update_json_hash(
        digest,
        {
            "schema_version": SCHEMA_VERSION,
            "model": getattr(model, "name", ""),
            "shape": list(expression.shape),
            "genes": [str(value) for value in expression.index],
            "cells": [str(value) for value in expression.columns],
            "parameters": {
                key: args.get(key)
                for key in FINGERPRINT_ARGUMENTS
                if key in args
            },
        },
    )

    model_reactions = []
    for reaction_id, reaction in sorted(
        getattr(model, "reactions", {}).items()
    ):
        association = getattr(reaction, "gene_associations", None)
        model_reactions.append(
            {
                "id": str(reaction_id),
                "bounds": [
                    float(getattr(reaction, "lower_bound", 0.0)),
                    float(getattr(reaction, "upper_bound", 0.0)),
                ],
                "reactants": sorted(getattr(reaction, "reactants", {}).items()),
                "products": sorted(getattr(reaction, "products", {}).items()),
                "gpr": (
                    association.to_serializable()
                    if association is not None
                    else None
                ),
            }
        )
    _update_json_hash(digest, model_reactions)

    for start in range(0, expression.shape[0], 256):
        stop = min(expression.shape[0], start + 256)
        values = np.ascontiguousarray(
            expression.iloc[start:stop].to_numpy(dtype="<f8")
        )
        digest.update(values.tobytes())
    return digest.hexdigest()


def _atomic_json_write(path, value):
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _as_feature_matrix(series_by_cell, feature_index):
    if not series_by_cell:
        return np.empty((0, len(feature_index)), dtype=float)
    return np.vstack(
        [
            series.reindex(feature_index).to_numpy(dtype=float)
            for series in series_by_cell
        ]
    )


class BatchCheckpoint:
    """Persist FBA output in atomic batches and load all completed cells."""

    def __init__(
        self,
        directory,
        fingerprint,
        batch_size=64,
        overwrite=False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.directory = Path(directory)
        self.fingerprint = str(fingerprint)
        self.batch_size = int(batch_size)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / MANIFEST_NAME

        if overwrite:
            self._clear_owned_files()

        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ScCompassError(
                    "Checkpoint schema is incompatible with this scCompass version"
                )
            if manifest.get("fingerprint") != self.fingerprint:
                raise CheckpointMismatchError(
                    "Checkpoint belongs to a different input or parameters; "
                    "choose a new checkpoint directory or overwrite it"
                )
        else:
            if self._batch_files():
                raise ScCompassError(
                    "Checkpoint batches exist without a manifest"
                )
            _atomic_json_write(
                self.manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": self.fingerprint,
                    "batch_size": self.batch_size,
                },
            )
        self._completed = {
            record["cell"] for record in self._load_records()
        } if self._batch_files() else set()

    def _batch_files(self):
        return sorted(self.directory.glob("batch-*.npz"))

    def _clear_owned_files(self):
        for path in self._batch_files():
            path.unlink()
        for path in self.directory.glob(".batch-*.tmp"):
            path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def write_batch(self, results):
        """Atomically write one batch of cell results."""

        results = list(results)
        if not results:
            return None
        if len(results) > self.batch_size:
            raise ValueError(
                f"Batch has {len(results)} cells but batch_size is {self.batch_size}"
            )
        cells = [str(result.cell) for result in results]
        if len(cells) != len(set(cells)):
            raise ValueError("A checkpoint batch cannot contain duplicate cells")
        if set(cells).intersection(self._completed):
            raise ValueError("A checkpoint batch contains already completed cells")

        reaction_features = sorted(
            {str(feature) for result in results for feature in result.reactions.index}
        )
        uptake_features = sorted(
            {
                str(feature)
                for result in results
                if result.uptake is not None
                for feature in result.uptake.index
            }
        )
        secretion_features = sorted(
            {
                str(feature)
                for result in results
                if result.secretion is not None
                for feature in result.secretion.index
            }
        )

        arrays = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "cells": np.asarray(cells, dtype=str),
            "reaction_features": np.asarray(reaction_features, dtype=str),
            "reaction_values": _as_feature_matrix(
                [result.reactions for result in results], reaction_features
            ),
            "uptake_features": np.asarray(uptake_features, dtype=str),
            "uptake_values": _as_feature_matrix(
                [
                    result.uptake
                    if result.uptake is not None
                    else pd.Series(dtype=float)
                    for result in results
                ],
                uptake_features,
            ),
            "secretion_features": np.asarray(secretion_features, dtype=str),
            "secretion_values": _as_feature_matrix(
                [
                    result.secretion
                    if result.secretion is not None
                    else pd.Series(dtype=float)
                    for result in results
                ],
                secretion_features,
            ),
        }
        destination = self.directory / f"batch-{uuid4().hex}.npz"
        fd, temporary = tempfile.mkstemp(
            dir=self.directory, prefix=".batch-", suffix=".tmp"
        )
        os.close(fd)
        try:
            with open(temporary, "wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._completed.update(cells)
            return destination
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load_records(self):
        records = []
        seen = set()
        for path in self._batch_files():
            with np.load(path, allow_pickle=False) as batch:
                if int(batch["schema_version"]) != SCHEMA_VERSION:
                    raise ScCompassError(f"Unsupported checkpoint batch: {path}")
                cells = batch["cells"].astype(str).tolist()
                if len(cells) != batch["reaction_values"].shape[0]:
                    raise ScCompassError(f"Corrupt checkpoint batch: {path}")
                duplicate = seen.intersection(cells)
                if duplicate:
                    raise ScCompassError(
                        f"Duplicate completed cells in checkpoint: {sorted(duplicate)}"
                    )
                seen.update(cells)
                reaction_features = batch["reaction_features"].astype(str).tolist()
                uptake_features = batch["uptake_features"].astype(str).tolist()
                secretion_features = batch["secretion_features"].astype(str).tolist()
                for index, cell in enumerate(cells):
                    records.append(
                        {
                            "cell": cell,
                            "reactions": pd.Series(
                                batch["reaction_values"][index],
                                index=reaction_features,
                                name=cell,
                                dtype=float,
                            ),
                            "uptake": pd.Series(
                                batch["uptake_values"][index],
                                index=uptake_features,
                                name=cell,
                                dtype=float,
                            ),
                            "secretion": pd.Series(
                                batch["secretion_values"][index],
                                index=secretion_features,
                                name=cell,
                                dtype=float,
                            ),
                        }
                    )
        return records

    def completed_cells(self):
        """Return the set of cells represented in completed batches."""

        return set(self._completed)

    def load_frames(self, cell_order):
        """Collect all checkpoint batches into reaction and exchange matrices."""

        records = {record["cell"]: record for record in self._load_records()}
        ordered_cells = [str(cell) for cell in cell_order]
        missing = [cell for cell in ordered_cells if cell not in records]
        if missing:
            raise ScCompassError(
                f"Checkpoint is incomplete for {len(missing)} requested cells"
            )

        frames = {}
        for key in ("reactions", "uptake", "secretion"):
            values = [records[cell][key] for cell in ordered_cells]
            frame = pd.DataFrame.from_dict(
                {cell: records[cell][key] for cell in ordered_cells},
                orient="index",
            )
            frames[key] = frame.reindex(ordered_cells).sort_index(axis=1)
        return frames
