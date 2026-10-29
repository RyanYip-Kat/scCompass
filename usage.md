# scCompass Usage Guide

This guide documents the public Python API, result schema, plotting helpers,
checkpointing, parallel execution, and the PBMC pipeline.

## Contents

- [Installation](#installation)
- [Input data](#input-data)
- [Core analysis](#core-analysis)
- [Result AnnData](#result-anndata)
- [I/O and table export](#io-and-table-export)
- [Group statistics](#group-statistics)
- [Plotting](#plotting)
- [One-command report](#one-command-report)
- [Pipeline scripts](#pipeline-scripts)
- [Checkpoints and parallelism](#checkpoints-and-parallelism)
- [Solver backends](#solver-backends)
- [Interpretation and caveats](#interpretation-and-caveats)
- [Testing](#testing)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[plot]"
```

Add solver dependencies:

```bash
python -m pip install -e ".[plot,solver]"
```

CPLEX is proprietary and requires a valid license. The Community Edition is
not sufficient for full RECON2. HiGHS can be selected without CPLEX:

```python
result = scCompass.compass(adata, solver_backend="highs")
```

Optional cache location:

```bash
export SCCOMPASS_CACHE_DIR=/path/to/writable/cache
```

## Input data

```python
import anndata as ad
import scCompass

adata = ad.read_h5ad(
    "testdata/pbmc5k_querydata.h5ad",
    backed="r",
)
```

The example fixture contains:

```text
58,677 cells
33,421 genes
layers: counts, data
obsm: X_pca, X_umap
```

Use `layers["counts"]` rather than the pre-normalized `X` for this fixture:

```python
result = scCompass.compass(
    adata,
    expression_layer="counts",
)
```

No normalization is performed by `scCompass`. Input should be linear/count
scale because the original Compass penalty path applies its own
`log2(expression + 1)`.

If gene IDs are not in `var_names`:

```python
result = scCompass.compass(
    adata,
    expression_layer=None,
    gene_symbol_key="gene_symbol",
)
```

## Core analysis

### `scCompass.compass`

```python
result = scCompass.compass(
    adata,
    model="RECON2_mat",
    media="default-media",
    species="homo_sapiens",
    expression_layer="counts",
    gene_symbol_key=None,
    lambda_=0.0,
    num_neighbors=30,
    calc_metabolites=True,
    isoform_summing="remove-summing",
    and_function="mean",
    symmetric_kernel=False,
    penalty_diffusion="knn",
    penalty_backend="vectorized",
    cell_chunk_size=1024,
    checkpoint_dir=None,
    checkpoint_batch_size=64,
    overwrite_checkpoint=False,
    num_processes=1,
    num_threads=1,
    microcluster_size=None,
    microcluster_filter="fano",
    microcluster_filter_threshold=None,
    solver_backend="cplex",
    select_reactions=None,
    select_subsystems=None,
    test_mode=False,
    sample_size=None,
)
```

Key parameters:

| Parameter | Meaning |
|---|---|
| `expression_layer` | `"counts"`, another layer name, or `None` for `X` |
| `gene_symbol_key` | Gene-symbol column in `adata.var` |
| `lambda_` | Neighbor smoothing strength in `[0, 1]` |
| `and_function` | GPR AND aggregation: `mean`, `min`, or `median` |
| `penalty_backend` | `vectorized` or legacy per-cell calculation |
| `cell_chunk_size` | Vectorized GPR chunk size |
| `microcluster_size` | Target cells per microcluster |
| `num_processes` | Process-based FBA workers |
| `num_threads` | CPLEX threads within each worker |
| `checkpoint_dir` | Persistent resumable output directory |
| `solver_backend` | `cplex` or `highs` |
| `sample_size` | Bounded debugging subset |

### Short smoke run

```python
smoke = scCompass.compass(
    adata,
    expression_layer="counts",
    sample_size=100,
    test_mode=True,
    solver_backend="cplex",
)
```

`sample_size` is for tests only. It is not used by the full-data scripts.

### Full-data run

```python
result = scCompass.compass(
    adata,
    expression_layer="counts",
    penalty_backend="vectorized",
    solver_backend="cplex",
    num_processes=8,
    num_threads=1,
    microcluster_size=1000,
    microcluster_filter="novar",
    checkpoint_dir="outputs/pbmc5k_full_pipeline/checkpoint",
    checkpoint_batch_size=16,
)
```

## Result AnnData

The result preserves the complete input object:

```text
result.X
result.var
result.layers
result.raw
result.varm
result.varp
result.obs
result.obsm["X_pca"]
result.obsm["X_umap"]
result.obsp
result.uns["seurat2anndata"]
```

Compass outputs are added as:

```text
result.obsm["scCompass_reactions"]
result.obsm["scCompass_uptake"]
result.obsm["scCompass_secretion"]

result.uns["scCompass_reaction_columns"]
result.uns["scCompass_reaction_var"]
result.uns["scCompass_uptake_columns"]
result.uns["scCompass_secretion_columns"]
result.uns["scCompass"]
```

For the current full-data run:

```text
input shape:              58,677 × 33,421
reaction score shape:     58,677 × 230
uptake score shape:       58,677 × 81
secretion score shape:    58,677 × 81
microclusters:            61
```

Create the legacy score-only object:

```python
scores = scCompass.result.to_score_anndata(result)
```

## I/O and table export

```python
scCompass.io.save_h5ad(result, "outputs/pbmc_result.h5ad")

reactions = scCompass.to_dataframe(result, key="reactions")
uptake = scCompass.to_dataframe(result, key="uptake")
secretion = scCompass.to_dataframe(result, key="secretion")

scCompass.export_table(result, "outputs/reactions.tsv")
scCompass.export_table(result, "outputs/uptake.tsv", key="uptake")
```

`to_dataframe()` returns cells as rows and reactions/metabolites as columns.

## Group statistics

### `group_summary`

```python
summary = scCompass.tl.group_summary(
    result,
    groupby="celltype",
    key="reactions",
    include_exchange=False,
)
```

Output columns:

```text
group
feature
n
mean
median
std
fraction_positive
```

Use `key="uptake"` or `key="secretion"` for metabolites.

### `differential`

```python
difference = scCompass.tl.differential(
    result,
    groupby="celltype",
    key="reactions",
)
```

For more than two groups, it uses Kruskal-Wallis tests and BH correction. It
reports group means, best/worst group, effect size, and adjusted p-values.

Limit to selected features:

```python
difference = scCompass.tl.differential(
    result,
    groupby="celltype",
    features=["PFK_pos", "FBP_pos", "DPGase_pos"],
)
```

### `pathway_summary`

```python
pathways = scCompass.tl.pathway_summary(
    result,
    groupby="celltype",
)
```

Reaction penalties are averaged per cell within each annotated subsystem.
Exchange reactions are excluded by default.

### `metabolite_differential`

```python
metabolites = scCompass.tl.metabolite_differential(
    result,
    groupby="celltype",
    group_a="CD4 Naive",
    group_b="CD4 Tem",
    direction="both",
)
```

For each metabolite and pathway:

```text
Cohen's d = (mean_group_a - mean_group_b) / pooled_SD
p-value = Mann-Whitney U
FDR = Benjamini-Hochberg within direction/pathway panel
```

## Plotting

### Reaction UMAP

```python
scCompass.pl.umap(
    result,
    color=["PFK_pos", "FBP_pos", "DPGase_pos", "celltype"],
    ncols=2,
)
```

![Reaction UMAP](docs/images/reaction_umap.png)

One continuous score per panel is preferable. Use the cell-type panel as a
reference.

### Group heatmap

```python
scCompass.pl.heatmap(
    result,
    groupby="celltype",
    top_n=30,
    zscore=True,
)
```

![Reaction heatmap](docs/images/reaction_heatmap.png)

Default behavior:

- top differential reactions
- group mean matrix
- per-reaction z-score across groups
- `RdBu_r` diverging color map
- exchange reactions excluded unless explicitly supplied

For raw mean penalties:

```python
scCompass.pl.heatmap(
    result,
    groupby="celltype",
    zscore=False,
    cmap="viridis",
)
```

### Reaction raincloud plot

```python
scCompass.pl.raincloud(
    result,
    groupby="celltype",
    features=["DPGase_pos", "PFK_pos", "FBP_pos"],
    max_points_per_group=100,
)
```

![Reaction raincloud plot](docs/images/reaction_raincloud.png)

The helper uses violin + box + points when groups have at least 10 cells and
falls back to raw points for smaller groups. At most 100 points per group are
shown to control overplotting.

### Metabolite dot plot

```python
scCompass.pl.metabolite_dotplot(
    result,
    groupby="celltype",
    direction="both",
    top_n=20,
)
```

![Metabolite dot plot](docs/images/metabolite_dotplot.png)

- color = mean uptake/secretion penalty
- size = fraction of cells with non-zero score
- uptake and secretion remain separate

### Metabolite group-difference volcano

```python
scCompass.pl.metabolite_volcano(
    result,
    groupby="celltype",
    group_a="CD4 Naive",
    group_b="CD4 Tem",
    effect_threshold=0.2,
    fdr_threshold=0.05,
    y_cap=20,
)
```

![Metabolite volcano plot](docs/images/metabolite_volcano.png)

Each point is a metabolite. Panels are pathway × exchange direction. Positive x
means higher penalty in `group_a`; negative x means higher penalty in
`group_b`. The y-axis display cap prevents underflowed p-values from compressing
the rest of the data.

Change the contrast:

```python
scCompass.pl.metabolite_volcano(
    result,
    groupby="celltype",
    group_a="Naive BC",
    group_b="Memory BC",
)
```

### Local metabolic network

Select by metabolite:

```python
scCompass.pl.network(
    result,
    metabolite="lac_L[c]",
)
```

Select top differences:

```python
scCompass.pl.network(
    result,
    groupby="celltype",
    top_n=20,
)
```

![Metabolic network](docs/images/metabolic_network.png)

Square nodes are reactions and circular nodes are metabolites. Arrows follow
reactant → reaction → product. Reaction color represents mean penalty.

## One-command report

```python
artifacts = scCompass.run_report(
    result,
    "outputs/pbmc_report",
    groupby="celltype",
    top_n=30,
    include_exchange=False,
    volcano_groups=("CD4 Naive", "CD4 Tem"),
)
```

Generated files:

```text
reaction_group_summary.tsv
reaction_differential.tsv
pathway_summary.tsv
metabolite_differential.tsv
reaction_umap.png
reaction_heatmap.png
reaction_raincloud.png
metabolite_dotplot.png
metabolite_volcano.png
metabolic_network.png
```

Vector formats are available with `output_format="pdf"` or `"svg"`.

## Pipeline scripts

### Full calculation

```bash
python scripts/pipeline.py \
  --input testdata/pbmc5k_querydata.h5ad \
  --output-dir outputs/pbmc5k_full_pipeline \
  --model-scope central-carbon \
  --solver-backend cplex \
  --num-processes 8 \
  --num-threads 1 \
  --microcluster-size 1000 \
  --microcluster-filter novar \
  --checkpoint-batch-size 16
```

### Visualize an existing result

```bash
python scripts/pipeline_vis.py \
  --input outputs/pbmc5k_full_pipeline/result.h5ad \
  --output-dir outputs/pbmc5k_visual_report \
  --groupby celltype \
  --top-n 30 \
  --volcano-group-a "CD4 Naive" \
  --volcano-group-b "CD4 Tem"
```

Useful flags:

```text
--include-exchange
--export-scores
--format png|pdf|svg
--features R1,R2,R3
--effect-threshold 0.2
--fdr-threshold 0.05
--volcano-y-cap 20
```

### Combined workflow

```bash
bash scripts/run_pbmc5k.sh
```

Environment overrides:

```bash
FORCE=1
RESET_CHECKPOINT=1
EXPORT_SCORES=1
MODEL_SCOPE=full
SOLVER_BACKEND=cplex
NUM_PROCESSES=8
NUM_THREADS=1
MICROCLUSTER_SIZE=1000
MICROCLUSTER_FILTER=novar
GROUPBY=celltype
TOP_N=30
FIGURE_FORMAT=png
VOLCANO_GROUP_A='CD4 Naive'
VOLCANO_GROUP_B='CD4 Tem'
VOLCANO_Y_CAP=20
```

## Checkpoints and parallelism

```python
result = scCompass.compass(
    adata,
    checkpoint_dir="outputs/checkpoint",
    checkpoint_batch_size=64,
    num_processes=8,
    num_threads=1,
)
```

Checkpoint rules:

- Completed cells are skipped on resume.
- The fingerprint includes expression values, model structure, and
  solver-affecting settings.
- A changed input or setting raises `CheckpointMismatchError`.
- Each batch is written atomically as compressed NPZ.
- `overwrite_checkpoint=True` removes only scCompass-owned checkpoint files.

Each worker builds a fresh FBA problem per cell. This preserves the original
Compass per-cell numerical behavior.

## Solver backends

CPLEX:

```python
scCompass.compass(adata, solver_backend="cplex")
```

HiGHS:

```python
scCompass.compass(adata, solver_backend="highs")
```

HiGHS avoids the CPLEX Community Edition limit but is currently recommended
only for reduced models and small batches.

## Interpretation and caveats

- Scores are penalties, not measured fluxes.
- Lower scores are generally more favorable.
- Exact zero can represent either no penalty or a blocked reaction in the
  legacy algorithm.
- Compare a reaction against its own distribution rather than using one global
  score threshold across all reactions.
- Cell-level p-values can be anti-conservative when cells come from repeated
  donors. Use donor-level pseudobulk or mixed models for biological inference.
- Microclustering pools cells and therefore reduces single-cell resolution by
  design.
- Full RECON2 requires a full CPLEX license in the current implementation.

## Testing

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python -m pytest -q
```

The tests cover:

- bundled model and Resources loading without external Compass;
- AnnData layers, duplicate genes, and missing genes;
- vectorized and legacy reaction penalties;
- CPLEX and HiGHS parity on small models;
- parallel FBA and checkpoint recovery;
- microcluster expansion;
- input-preserving and score-only AnnData output;
- group statistics and all plotting helpers;
- full 58,677-cell pipeline and report generation.
