# scCompass AnnData Integration Design

## Goal

Provide an AnnData-native package, `scCompass`, around the existing COMAPSS
algorithm. The conversion must preserve reaction-penalty and flux-score
semantics, preserve visualization inputs, and write a single result AnnData
that can be used directly for plotting and tabular export.

## Compatibility Boundary

The existing implementation under `compass/` is the numerical reference. The
first release may make the following behavior-preserving changes:

- replace removed pandas APIs (`iteritems`, `applymap`, and `DataFrame.append`)
  with their current equivalents;
- defer importing optional or solver-specific dependencies until use;
- avoid reading unrelated expression genes;
- replace hard-coded and package-internal writable paths with configuration;
- turn silent collection failures into explicit errors;
- expose a result factory without changing score formulas.

The following remain semantically unchanged:

- reaction-expression and penalty formulas;
- the smoothing parameter and KNN/PCA behavior;
- FBA constraints, objective, bounds, beta, and exchange handling;
- reaction and metabolite identifiers;
- random seeds.

Risky optimizations such as a new solver, vectorized penalties, or sparse FBA
are opt-in backends. They are not part of the default numerical path.

## Input Contract

`scCompass.compass()` accepts an `anndata.AnnData` object.

- `obs_names` are cell identifiers and their order is preserved.
- Genes come from `var_names`, or from `var[gene_symbol_key]` when specified.
- `expression_layer="counts"` selects `layers["counts"]`.
- `expression_layer=None` selects `X`.
- No normalization is applied implicitly.
- Symbols are uppercased and duplicate symbols are summed as required by the
  legacy DataFrame path.
- Only genes that can match the metabolic model and its aliases are sliced and
  converted for the core.
- The adapter preserves legacy missing-gene behavior rather than imputing
  expression.
- The core receives a gene-by-cell `pandas.DataFrame`.

## Result Contract

The primary result is an `AnnData` with the same cell order and visualization
data as the input:

```text
result.X       cells x reactions
result.obs     copied from input
result.var     reaction identifiers and metadata
result.obsm    input embeddings plus optional uptake/secretion matrices
result.obsp    copied from input
result.uns     copied from input plus namespaced scCompass metadata
```

Reaction scores in `result.X` are deliberate. Existing COMAPSS plotting code
that treats `AnnData.X` as reaction scores continues to work. Uptake and
secretion use `obsm["scCompass_uptake"]` and
`obsm["scCompass_secretion"]`. Original expression is not placed in `X`;
optional input preservation may use `raw`.

`uns["scCompass"]` records the model, media, species, selected layer, core
parameters, mapped/missing genes, solver backend, version, and output keys.

## Public API

```python
import scCompass as scc

result = scc.compass(
    adata,
    model="RECON2_mat",
    media="default-media",
    species="homo_sapiens",
    expression_layer="counts",
    gene_symbol_key=None,
    lambda_=0.0,
    num_neighbors=30,
    calc_metabolites=True,
)
```

Supporting APIs:

```python
scc.to_dataframe(result, key="reactions")
scc.io.export_table(result, path)
scc.pl.umap(result, color=[reaction_id])
```

## Numerical and Visual Parity

Parity is checked at three levels:

1. Numeric: shared reaction/metabolite values match the legacy pipeline within
   `1e-8`; exact-zero reactions remain exact zero.
2. Structural: cell identifiers, reaction ordering, embeddings, and graph
   coordinates match the input or legacy output.
3. Visual: plots from legacy result tables and from result AnnData use the same
   values, labels, embeddings, color limits, and plotting parameters. Rendering
   under a fixed backend, DPI, and size must produce equivalent images.

The current `compass/` snapshot has no plotting implementation. Compatibility
therefore means preserving every data field needed by external plotting code.
If a plotting implementation is added later, it consumes the result AnnData
without changing those fields.

## Error Handling

- Invalid expression layers and gene keys fail with actionable `ValueError`.
- Duplicate non-unique cell identifiers fail before computation.
- Missing model genes are reported and counted, then follow legacy behavior.
- Solver import/license failures identify the solver requirement explicitly.
- Result collisions never silently write partial values.

## Testing

Tests cover in-memory AnnData validation, layer selection, symbol mapping,
duplicate genes, missing genes, result schemas, h5ad round trips, table and plot
access, and the PBMC test fixture. The full-size 58,677-cell fixture is exercised
through a backed, bounded slice in the default suite. A full FBA run requires
CPLEX and is a separate slow integration test.

## Non-goals

- Reimplementing Compass with a different optimization model.
- Automatically normalizing single-cell expression.
- Preserving input gene expression inside result `X`.
- Making a new solver the default before numerical parity is established.
