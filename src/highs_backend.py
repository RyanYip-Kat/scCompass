"""Optional SciPy/HiGHS implementation of Compass cell scoring."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .core.globals import BETA, EXCHANGE_LIMIT
from .exceptions import SolverUnavailableError


def _selected_reaction_ids(args, model):
    selected_reactions = set()
    selected_subsystems = set()
    if args.get("select_reactions"):
        with open(args["select_reactions"]) as handle:
            selected_reactions.update(
                line.strip() for line in handle if line.strip()
            )
    if args.get("select_subsystems"):
        with open(args["select_subsystems"]) as handle:
            selected_subsystems.update(
                line.strip() for line in handle if line.strip()
            )
    for reaction in model.reactions.values():
        if reaction.subsystem in selected_subsystems:
            selected_reactions.add(reaction.id)
    if not selected_reactions:
        return None
    return selected_reactions


class _HighsProblem:
    def __init__(self, model):
        self.model = model
        self.names = list(model.reactions)
        self.index = {name: position for position, name in enumerate(self.names)}
        self.lower = np.asarray(
            [reaction.lower_bound for reaction in model.reactions.values()],
            dtype=float,
        )
        self.upper = np.asarray(
            [reaction.upper_bound for reaction in model.reactions.values()],
            dtype=float,
        )

        reaction_rows = []
        reaction_columns = []
        reaction_values = []
        self.row_names = []
        self.row_index = {}
        for metabolite in model.species.values():
            entries = []
            for reaction in model.reactions.values():
                coefficient = 0.0
                if metabolite.id in reaction.reactants:
                    coefficient -= reaction.reactants[metabolite.id]
                if metabolite.id in reaction.products:
                    coefficient += reaction.products[metabolite.id]
                if coefficient:
                    entries.append(
                        (
                            self.index[reaction.id],
                            float(coefficient),
                        )
                    )
            if entries:
                row = len(self.row_names)
                self.row_index[metabolite.id] = row
                self.row_names.append(metabolite.id)
                for column, value in entries:
                    reaction_rows.append(row)
                    reaction_columns.append(column)
                    reaction_values.append(value)
        self.eq_matrix = sparse.coo_matrix(
            (reaction_values, (reaction_rows, reaction_columns)),
            shape=(len(self.row_names), len(self.names)),
        ).tocsr()

    def set_upper_bound(self, name, value):
        self.upper[self.index[name]] = float(value)

    def upper_bound(self, name):
        return float(self.upper[self.index[name]])

    def lower_bound(self, name):
        return float(self.lower[self.index[name]])

    def add_variable(self, name, lower, upper, row_name, coefficient):
        self.index[name] = len(self.names)
        self.names.append(name)
        self.lower = np.append(self.lower, float(lower))
        self.upper = np.append(self.upper, float(upper))
        column = np.zeros(self.eq_matrix.shape[0], dtype=float)
        if row_name in self.row_index:
            column[self.row_index[row_name]] = float(coefficient)
        self.eq_matrix = sparse.hstack(
            [self.eq_matrix, sparse.csr_matrix(column[:, None])],
            format="csr",
        )

    def remove_variable(self, name):
        position = self.index[name]
        keep = np.ones(len(self.names), dtype=bool)
        keep[position] = False
        self.eq_matrix = self.eq_matrix[:, keep].tocsr()
        self.names.pop(position)
        self.lower = self.lower[keep]
        self.upper = self.upper[keep]
        self.index = {
            variable: index for index, variable in enumerate(self.names)
        }

    def _objective(self, penalties, maximize_name=None):
        objective = np.zeros(len(self.names), dtype=float)
        if maximize_name is not None:
            objective[self.index[maximize_name]] = 1.0
            return -objective
        for name, value in penalties.items():
            if name in self.index:
                objective[self.index[name]] = float(value)
        return objective

    def solve(self, penalties, maximize_name=None, lower_constraint=None):
        objective = self._objective(penalties, maximize_name=maximize_name)
        bounds = list(zip(self.lower, self.upper))
        a_ub = b_ub = None
        if lower_constraint is not None:
            name, lower = lower_constraint
            row = np.zeros(len(self.names), dtype=float)
            row[self.index[name]] = -1.0
            a_ub = row[None, :]
            b_ub = np.asarray([-float(lower)])
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=self.eq_matrix,
            b_eq=np.zeros(self.eq_matrix.shape[0]),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise SolverUnavailableError(
                f"HiGHS failed to solve Compass LP: {result.message}"
            )
        value = -float(result.fun) if maximize_name else float(result.fun)
        if abs(value) < 1e-12:
            value = 0.0
        return value


def _compass_reactions(problem, model, penalties, args):
    scores = {}
    reactions = list(model.reactions.values())
    if args.get("test_mode"):
        reactions = reactions[:100]
    selected = _selected_reaction_ids(args, model)
    if selected is not None:
        reactions = [
            reaction
            for reaction in reactions
            if reaction.id in selected or reaction.id[:-4] in selected
        ]

    for reaction in reactions:
        if reaction.is_exchange:
            continue
        partner = reaction.reverse_reaction
        if partner is not None:
            old_upper = problem.upper_bound(partner.id)
            problem.set_upper_bound(
                partner.id, max(problem.lower_bound(partner.id), 0.0)
            )
        maximum = problem.solve(penalties, maximize_name=reaction.id)
        if maximum == 0:
            scores[reaction.id] = 0.0
        else:
            scores[reaction.id] = problem.solve(
                penalties,
                lower_constraint=(reaction.id, BETA * maximum),
            )
        if partner is not None:
            problem.set_upper_bound(partner.id, old_upper)
    return scores


def _compass_exchange(problem, model, penalties, only_exchange, args):
    uptake_scores = {}
    secretion_scores = {}
    exchange_scores = {}
    metabolites = list(model.species.values())
    if args.get("test_mode"):
        metabolites = metabolites[:50]
    selected = _selected_reaction_ids(args, model)

    for metabolite in metabolites:
        metabolite_id = metabolite.id
        if metabolite_id not in problem.row_index:
            uptake_scores[metabolite_id] = 0.0
            secretion_scores[metabolite_id] = 0.0
            continue

        reactions = [
            reaction
            for reaction in model.reactions.values()
            if metabolite_id in reaction.reactants
            or metabolite_id in reaction.products
        ]
        if only_exchange:
            reactions = [reaction for reaction in reactions if reaction.is_exchange]
        if selected is not None:
            reactions = [
                reaction
                for reaction in reactions
                if reaction.id in selected or reaction.id[:-4] in selected
            ]

        uptake_reaction = None
        secretion_reaction = None
        extra_uptake = []
        extra_secretion = []
        for reaction in reactions:
            if reaction.is_exchange and metabolite_id in reaction.products:
                if uptake_reaction is None:
                    uptake_reaction = reaction.id
                else:
                    extra_uptake.append(reaction.id)
            elif reaction.is_exchange and metabolite_id in reaction.reactants:
                if secretion_reaction is None:
                    secretion_reaction = reaction.id
                else:
                    extra_secretion.append(reaction.id)

        if only_exchange and uptake_reaction is None and secretion_reaction is None:
            continue

        added_uptake = uptake_reaction is None
        added_secretion = secretion_reaction is None
        if added_secretion:
            secretion_reaction = metabolite_id + "_SECRETION"
            problem.add_variable(
                secretion_reaction,
                0.0,
                model.maximum_flux,
                metabolite_id,
                -1.0,
            )
        if added_uptake:
            uptake_reaction = metabolite_id + "_UPTAKE"
            problem.add_variable(
                uptake_reaction,
                0.0,
                EXCHANGE_LIMIT,
                metabolite_id,
                1.0,
            )

        all_uptake = [uptake_reaction, *extra_uptake]
        all_secretion = [secretion_reaction, *extra_secretion]

        old_uptake = {}
        for name in all_uptake:
            old_uptake[name] = problem.upper_bound(name)
            problem.set_upper_bound(name, max(problem.lower_bound(name), 0.0))
        old_extra_secretion = {}
        for name in extra_secretion:
            old_extra_secretion[name] = problem.upper_bound(name)
            problem.set_upper_bound(name, max(problem.lower_bound(name), 0.0))
        secretion_max = problem.solve(
            penalties, maximize_name=secretion_reaction
        )
        secretion_scores[metabolite_id] = problem.solve(
            penalties,
            lower_constraint=(secretion_reaction, BETA * secretion_max),
        )
        for name, value in old_uptake.items():
            problem.set_upper_bound(name, value)
        for name, value in old_extra_secretion.items():
            problem.set_upper_bound(name, value)

        old_extra_uptake = {}
        for name in extra_uptake:
            old_extra_uptake[name] = problem.upper_bound(name)
            problem.set_upper_bound(name, max(problem.lower_bound(name), 0.0))
        old_secretion = {}
        for name in all_secretion:
            old_secretion[name] = problem.upper_bound(name)
            problem.set_upper_bound(name, max(problem.lower_bound(name), 0.0))
        uptake_max = problem.solve(penalties, maximize_name=uptake_reaction)
        uptake_scores[metabolite_id] = problem.solve(
            penalties,
            lower_constraint=(uptake_reaction, BETA * uptake_max),
        )
        for name, value in old_extra_uptake.items():
            problem.set_upper_bound(name, value)
        for name, value in old_secretion.items():
            problem.set_upper_bound(name, value)

        if added_uptake:
            problem.remove_variable(uptake_reaction)
        else:
            for name in all_uptake:
                exchange_scores[name] = uptake_scores[metabolite_id]
        if added_secretion:
            problem.remove_variable(secretion_reaction)
        else:
            for name in all_secretion:
                exchange_scores[name] = secretion_scores[metabolite_id]

    return uptake_scores, secretion_scores, exchange_scores


def solve_cell(model, penalties, args):
    """Return Compass reaction, uptake, and secretion scores using HiGHS."""

    problem = _HighsProblem(model)
    reactions = _compass_reactions(problem, model, penalties, args)
    uptake, secretion, exchange = _compass_exchange(
        problem,
        model,
        penalties,
        only_exchange=not args["calc_metabolites"],
        args=args,
    )
    reactions.update(exchange)
    return reactions, uptake, secretion
