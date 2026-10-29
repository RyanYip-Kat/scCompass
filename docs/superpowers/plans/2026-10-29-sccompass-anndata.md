# scCompass AnnData Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AnnData-native `scCompass` package around the existing COMAPSS implementation while preserving score and visualization semantics.

**Architecture:** A compatibility layer extracts only metabolic-model genes from AnnData and supplies the legacy gene-by-cell DataFrame contract. Legacy scoring is invoked behind a runner adapter, then converted into a score-centric result AnnData with copied cell metadata and embeddings.

**Tech Stack:** Python 3.12, AnnData, pandas, NumPy, SciPy, Matplotlib, optional CPLEX and libsbml, pytest

**Spec:** `docs/superpowers/specs/2026-10-29-sccompass-anndata-design.md`

## Global Constraints

- The default numerical path must preserve COMPASS formulas, parameters, identifiers, and seeds.
- `result.X` is a cells-by-reactions score matrix.
- `result.obs_names` order must exactly equal the input order.
- `result.obsm` and `result.obsp` visualization data must be copied unchanged from input.
- No expression normalization may happen implicitly.
- `layers["counts"]` is preferred only when explicitly selected or by documented default.
- Real end-to-end FBA requires CPLEX and its license.
- This workspace has no valid Git repository; do not initialize one or create commits.

---

### Task 1: Package and Public API

**Files:**
- Create: `tests/test_public_api.py`
- Create: `scCompass/__init__.py`
- Create: `scCompass/tl.py`
- Create: `scCompass/exceptions.py`
- Modify: `setup.py`

**Interfaces:**
- Consumes: none.
- Produces: `scCompass.compass`, `scCompass.tl.compass`, and `scCompass.__version__`.

- [ ] **Step 1: Write the failing public API test**

```python
import inspect
import scCompass


def test_public_compass_api_is_available():
    assert callable(scCompass.compass)
    assert scCompass.compass is scCompass.tl.compass
    params = inspect.signature(scCompass.compass).parameters
    assert "adata" in params
    assert "expression_layer" in params
    assert "lambda_" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_public_api.py -q`

Expected: FAIL because `scCompass` does not exist.

- [ ] **Step 3: Add package API**

Expose `tl` and top-level `compass`. Use a `NotImplementedError` only until the
runner task introduces the real implementation.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_public_api.py -q`

Expected: PASS.

### Task 2: AnnData Validation and Adapter

**Files:**
- Create: `tests/test_adata_adapter.py`
- Create: `scCompass/adata.py`

**Interfaces:**

```python
prepare_expression(
    adata,
    expression_layer="counts",
    gene_symbol_key=None,
    model_genes=None,
    sample_size=None,
) -> PreparedExpression
```

`PreparedExpression` contains `expression` (gene-by-cell DataFrame), `symbols`,
`mapped_symbols`, `missing_model_genes`, and metadata.

- [ ] **Step 1: Write failing adapter tests**

```python
import anndata as ad
import numpy as np
from scipy import sparse
from scCompass.adata import prepare_expression


def test_adapter_prefers_counts_and_keeps_cell_order():
    X = sparse.csr_matrix(np.log1p([[1, 2], [3, 4]]))
    layers = {"counts": sparse.csr_matrix([[1, 2], [3, 4]])}
    adata = ad.AnnData(X=X, layers=layers)
    adata.obs_names = ["cell_b", "cell_a"]
    adata.var_names = ["g1", "g2"]
    prepared = prepare_expression(
        adata, model_genes={"G1"}, sample_size=1
    )
    assert prepared.expression.shape == (1, 1)
    assert prepared.expression.columns.tolist() == ["cell_b"]
    assert prepared.expression.iloc[0, 0] == 1


def test_adapter_sums_duplicate_symbols():
    adata = ad.AnnData(
        X=sparse.csr_matrix([[1], [2], [3], [4]]),
        var={"symbol": ["A", "A", "B", "B"]},
        obs={"cell": ["b", "a", "b", "a"]},
    )
    prepared = prepare_expression(
        adata,
        expression_layer=None,
        gene_symbol_key="symbol",
        model_genes={"B"},
    )
    assert prepared.expression.shape == (1, 1)
    assert prepared.expression.loc["B"].tolist() == [7]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_adata_adapter.py -q`

Expected: FAIL because the adapter is missing.

- [ ] **Step 3: Implement the minimal adapter**

Validate layers, preserve obs order, uppercase symbols, aggregate duplicates,
filter model genes, and return a gene-by-cell DataFrame.

- [ ] **Step 4: Run adapter tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_adata_adapter.py -q`

Expected: PASS.

### Task 3: Result AnnData and Tabular APIs

**Files:**
- Create: `tests/test_results.py`
- Create: `scCompass/result.py`
- Create: `scCompass/io.py`

**Interfaces:**

```python
build_result(
    source,
    reaction_scores,
    reaction_metadata=None,
    uptake=None,
    secretion=None,
    run_metadata=None,
) -> anndata.AnnData
```

- [ ] **Step 1: Write failing result tests**

```python
import anndata as ad
import numpy as np
import pandas as pd
from scCompass.result import build_result
from scCompass.io import to_dataframe


def test_result_uses_cells_by_reactions_and_copies_visual_inputs():
    source = ad.AnnData(
        X=np.zeros((2, 3)),
        obs=pd.DataFrame({"celltype": ["T", "B"]}, index=["c1", "c2"]),
    )
    source.obsm["X_umap"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    scores = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=["c1", "c2"],
        columns=["R1", "R2"],
    )
    result = build_result(source, scores)
    assert result.shape == (2, 2)
    assert result.obs_names.tolist() == ["c1", "c2"]
    assert result.var_names.tolist() == ["R1", "R2"]
    np.testing.assert_array_equal(result.obsm["X_umap"], source.obsm["X_umap"])
    assert to_dataframe(result).index.tolist() == ["c1", "c2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_results.py -q`

Expected: FAIL because result construction is missing.

- [ ] **Step 3: Implement result and table helpers**

Copy `obs`, `obsm`, `obsp`, and `uns`; transpose reaction-by-cell inputs into
the cells-by-reactions contract; add reaction metadata and namespaced run data.

- [ ] **Step 4: Run result tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_results.py -q`

Expected: PASS.

### Task 4: Visualization Access

**Files:**
- Create: `tests/test_plotting.py`
- Create: `scCompass/pl.py`

**Interfaces:**

```python
umap(result, color, basis="X_umap", **kwargs)
```

- [ ] **Step 1: Write the failing plot test**

Use the Matplotlib `Agg` backend, build a two-cell result with `X_umap`, call
`scCompass.pl.umap`, and assert that a non-empty Matplotlib figure is returned.

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg PYTHONPATH=. .venv/bin/python -m pytest tests/test_plotting.py -q`

Expected: FAIL because `scCompass.pl` is missing.

- [ ] **Step 3: Implement the minimal wrapper**

Delegate to Scanpy when available, otherwise raise an actionable dependency
error. Do not recompute embeddings.

- [ ] **Step 4: Run plot tests**

Run: `MPLBACKEND=Agg PYTHONPATH=. .venv/bin/python -m pytest tests/test_plotting.py -q`

Expected: PASS.

### Task 5: Existing Core Compatibility

**Files:**
- Create: `tests/test_core_compat.py`
- Create: `scCompass/compat.py`
- Modify: `compass/compass/penalties.py`
- Modify: `compass/compass/algorithm.py`
- Modify: `compass/models/__init__.py`
- Modify: `compass/compass/cache.py`
- Modify: `compass/utils.py`

**Interfaces:**

Compatibility helpers preserve the behavior of removed pandas methods and make
solver/model imports lazy.

- [ ] **Step 1: Write failing compatibility tests**

Assert that a DataFrame can pass through the compatibility iterator and that
importing `compass.models` succeeds without loading `libsbml`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_core_compat.py -q`

Expected: FAIL on the missing helper or eager import.

- [ ] **Step 3: Apply behavior-preserving compatibility changes**

Replace removed pandas calls, defer optional imports, and use an external cache
directory when the package resource directory is not writable.

- [ ] **Step 4: Run compatibility and existing model tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_core_compat.py tests/test_models.py -q`

Expected: PASS when model resources and libsbml are available.

### Task 6: Runner Integration

**Files:**
- Create: `tests/test_runner.py`
- Create: `scCompass/_compat_core.py`
- Create: `scCompass/runner.py`
- Modify: `scCompass/tl.py`

**Interfaces:**

```python
run_compass(
    adata,
    core_runner=legacy_core_runner,
    ...
) -> anndata.AnnData
```

`legacy_core_runner(expression, model, media, species, args)` returns reaction,
uptake, and secretion DataFrames with cells as columns.

- [ ] **Step 1: Write a failing orchestration test**

Inject a deterministic fake core runner, provide a small AnnData, and assert
that the returned object is a result AnnData with expected values.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_runner.py -q`

Expected: FAIL because the runner is missing.

- [ ] **Step 3: Implement orchestration with dependency injection**

Map public arguments to the legacy core contract and inject the real adapter in
production. Raise an actionable solver error when CPLEX is unavailable.

- [ ] **Step 4: Run runner tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_runner.py -q`

Expected: PASS.

### Task 7: PBMC AnnData End-to-End Test

**Files:**
- Create: `tests/test_pbmc_end_to_end.py`

**Interfaces:**

The test reads `testdata/pbmc5k_querydata.h5ad` with `backed="r"`, creates a
bounded in-memory slice, runs the package with a dependency-injected reference
runner when the real solver is unavailable, writes h5ad, reloads it, exports a
table, and renders a UMAP.

- [ ] **Step 1: Write the end-to-end test**

Validate counts-layer selection, observed cell order, score shape, result
schema, h5ad round trip, table export, and non-empty plotting output.

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg PYTHONPATH=. .venv/bin/python -m pytest tests/test_pbmc_end_to_end.py -q`

Expected: FAIL until all integration APIs exist.

- [ ] **Step 3: Run the complete implementation**

Run:

```bash
MPLBACKEND=Agg PYTHONPATH=. .venv/bin/python -m pytest -q
```

Expected: all non-solver tests pass.

- [ ] **Step 4: Attempt the real solver test**

Run the same test with the real core when CPLEX, libsbml, and model resources
are available. Record the solver limitation if they are not.

### Task 8: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**

Document installation, AnnData input requirements, result schema, plot/table
examples, CPLEX requirements, and the bounded PBMC test command.

- [ ] **Step 1: Add usage and schema documentation**
- [ ] **Step 2: Run the documented smoke command**
- [ ] **Step 3: Verify all tests once more**

Run: `MPLBACKEND=Agg PYTHONPATH=. .venv/bin/python -m pytest -q`

Expected: all available tests pass.
