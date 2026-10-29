import numpy as np
import pandas as pd

from scCompass.core.compass.penalties import (
    eval_reaction_expression_single,
    eval_reaction_penalties_shared,
)
from scCompass.core.models.MetabolicModel import (
    Association,
    Gene,
    MetabolicModel,
    Reaction,
)
from scCompass.penalties import (
    eval_reaction_expression_vectorized,
    eval_reaction_penalties_vectorized,
)


def _gene(name, aliases=()):
    gene = Gene()
    gene.id = name
    gene.name = name
    gene.alt_symbols = list(aliases)
    return gene


def _reaction(reaction_id, association):
    reaction = Reaction()
    reaction.id = reaction_id
    reaction.lower_bound = 0.0
    reaction.upper_bound = 10.0
    reaction.gene_associations = association
    return reaction


def _association(kind, children):
    association = Association()
    association.type = kind
    association.children = children
    return association


def _gene_node(gene):
    association = Association()
    association.type = "gene"
    association.gene = gene
    return association


def _parity_model():
    g1 = _gene("G1")
    g2 = _gene("G2")
    missing = _gene("G3")
    alias_only = _gene("G4", aliases=["ALIAS1"])

    reactions = [
        _reaction("R_GENE", _gene_node(g1)),
        _reaction(
            "R_AND_MEAN",
            _association("and", [_gene_node(g1), _gene_node(g2)]),
        ),
        _reaction(
            "R_AND_MIN",
            _association("and", [_gene_node(g1), _gene_node(missing)]),
        ),
        _reaction(
            "R_OR",
            _association("or", [_gene_node(g1), _gene_node(missing)]),
        ),
        _reaction("R_ALIAS_ONLY", _gene_node(alias_only)),
        _reaction("R_NO_GPR", None),
    ]
    model = MetabolicModel("PenaltyParityModel")
    model.reactions = {reaction.id: reaction for reaction in reactions}
    return model


def _expression():
    return pd.DataFrame(
        {
            "cell1": [1.0, 8.0, 3.0, 5.0, 2.0],
            "cell2": [4.0, 1.0, 2.0, 0.0, 7.0],
            "cell3": [0.0, 0.0, 6.0, 2.0, 1.0],
        },
        index=["G1", "G2", "ALIAS1", "OTHER", "G5"],
    )


def test_vectorized_reaction_expression_matches_legacy_for_all_gpr_types():
    model = _parity_model()
    expression = _expression()

    vectorized = eval_reaction_expression_vectorized(
        model, expression, and_function="mean", cell_chunk_size=2
    )
    for cell in expression.columns:
        legacy = eval_reaction_expression_single(
            model, expression[cell], and_function="mean"
        )
        np.testing.assert_allclose(
            vectorized[cell],
            legacy.reindex(vectorized.index),
            equal_nan=True,
        )

    vectorized_min = eval_reaction_expression_vectorized(
        model, expression, and_function="min"
    )
    legacy_min = eval_reaction_expression_single(
        model, expression["cell1"], and_function="min"
    )
    np.testing.assert_allclose(
        vectorized_min["cell1"],
        legacy_min.reindex(vectorized_min.index),
        equal_nan=True,
    )

    vectorized_median = eval_reaction_expression_vectorized(
        model, expression, and_function="median"
    )
    for cell in expression.columns:
        legacy_median = eval_reaction_expression_single(
            model, expression[cell], and_function="median"
        )
        np.testing.assert_allclose(
            vectorized_median[cell],
            legacy_median.reindex(vectorized_median.index),
            equal_nan=True,
        )


def test_vectorized_penalties_match_legacy_without_diffusion():
    model = _parity_model()
    expression = _expression()
    lambda_ = 0.0

    vectorized = eval_reaction_penalties_vectorized(
        model,
        expression,
        lambda_=lambda_,
        num_neighbors=2,
        symmetric_kernel=False,
        and_function="mean",
        penalty_diffusion_mode="knn",
        cell_chunk_size=2,
    )
    legacy = eval_reaction_penalties_shared(
        model,
        expression,
        lambda_=lambda_,
        num_neighbors=2,
        symmetric_kernel=False,
        and_function="mean",
        penalty_diffusion_mode="knn",
    )

    pd.testing.assert_frame_equal(vectorized, legacy, check_exact=False, atol=1e-14)


def test_vectorized_penalties_match_legacy_with_weights():
    model = _parity_model()
    expression = _expression()
    weights = pd.DataFrame(
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
        index=expression.columns,
        columns=expression.columns,
    )

    vectorized = eval_reaction_penalties_vectorized(
        model,
        expression,
        lambda_=0.5,
        num_neighbors=2,
        symmetric_kernel=False,
        and_function="mean",
        penalty_diffusion_mode="knn",
        input_weights=weights,
    )
    legacy = eval_reaction_penalties_shared(
        model,
        expression,
        lambda_=0.5,
        num_neighbors=2,
        symmetric_kernel=False,
        and_function="mean",
        penalty_diffusion_mode="knn",
        input_weights=weights,
    )

    pd.testing.assert_frame_equal(vectorized, legacy, check_exact=False, atol=1e-14)


def test_vectorized_penalties_match_legacy_knn_diffusion():
    model = _parity_model()
    expression = pd.DataFrame(
        [
            [1.0, 4.0, 0.0, 2.0],
            [8.0, 1.0, 0.0, 4.0],
            [3.0, 2.0, 6.0, 1.0],
            [5.0, 0.0, 2.0, 7.0],
            [2.0, 7.0, 1.0, 3.0],
        ],
        index=["G1", "G2", "ALIAS1", "OTHER", "G5"],
        columns=["c1", "c2", "c3", "c4"],
    )

    vectorized = eval_reaction_penalties_vectorized(
        model,
        expression,
        lambda_=0.5,
        num_neighbors=2,
        symmetric_kernel=False,
        and_function="mean",
        penalty_diffusion_mode="knn",
    )
    legacy = eval_reaction_penalties_shared(
        model,
        expression,
        lambda_=0.5,
        num_neighbors=2,
        symmetric_kernel=False,
        and_function="mean",
        penalty_diffusion_mode="knn",
    )

    pd.testing.assert_frame_equal(vectorized, legacy, check_exact=False, atol=1e-14)
