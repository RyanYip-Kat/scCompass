"""Reusable metabolic submodels derived from the bundled Compass resources."""

from copy import deepcopy
from pathlib import Path

from .core import models
from .core.globals import RESOURCE_DIR
from .core.models.MetabolicModel import MetabolicModel
from .core.models.importCommon import clean_reactions, limit_maximum_flux


def build_central_carbon_submodel(model_name="RECON2_mat", species="homo_sapiens"):
    """Build a RECON2 central-carbon submodel compatible with CPLEX Community."""

    full_model = models.load_metabolic_model(model_name, species=species)
    central_path = (
        Path(RESOURCE_DIR) / "Recon2_export" / "central_carbon_rxns.txt"
    )
    central_ids = {
        line.strip() for line in central_path.read_text().splitlines() if line.strip()
    }
    selected = [
        reaction
        for reaction_id, reaction in full_model.reactions.items()
        if reaction_id in central_ids
    ]
    selected = [
        reaction for reaction in selected if not reaction.is_exchange
    ]
    if not selected:
        raise ValueError("No central-carbon reactions were found in the model")

    reactions = {}
    species_ids = set()
    for reaction in selected:
        reactions[reaction.id] = deepcopy(reaction)
        species_ids.update(reaction.reactants)
        species_ids.update(reaction.products)

    # RECON2 exchange reactions terminate on extracellular compartments and
    # cannot close a central-carbon-only network. Add explicit uptake/secretion
    # boundaries for every retained metabolite while preserving reaction GPRs.
    from .core.models.MetabolicModel import Reaction

    for index, metabolite_id in enumerate(sorted(species_ids)):
        uptake = Reaction()
        uptake.id = f"EXC_{index}_UPTAKE"
        uptake.lower_bound = 0.0
        uptake.upper_bound = 1.0
        uptake.products[metabolite_id] = 1.0
        secretion = Reaction()
        secretion.id = f"EXC_{index}_SECRETION"
        secretion.lower_bound = 0.0
        secretion.upper_bound = 1000.0
        secretion.reactants[metabolite_id] = 1.0
        reactions[uptake.id] = uptake
        reactions[secretion.id] = secretion

    submodel = MetabolicModel("RECON2_CENTRAL_CARBON")
    submodel.reactions = reactions
    submodel.species = {
        metabolite_id: deepcopy(full_model.species[metabolite_id])
        for metabolite_id in species_ids
    }
    submodel.compartments = deepcopy(full_model.compartments)
    submodel.media = "NoMedia"
    clean_reactions(submodel)
    submodel.limitExchangeReactions(limit=1.0)
    limit_maximum_flux(submodel, 1000)
    submodel.make_unidirectional()
    return submodel
