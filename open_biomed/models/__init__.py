from open_biomed.models.foundation_models.molt5 import MolT5
from open_biomed.models.foundation_models.biot5 import BioT5
from open_biomed.models.foundation_models.biot5_plus import BioT5_PLUS
from open_biomed.models.foundation_models.moleculestm import MoleculeSTM
# from open_biomed.models.molecule.graphmvp import GraphMVP
# from open_biomed.models.foundation_models.pharmolix_fm import PharmolixFM
# from open_biomed.models.protein.mutaplm.mutaplm import MutaPLM

MODEL_REGISTRY = {
    "text_based_molecule_editing": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS,
        "moleculestm": MoleculeSTM
    },
    "molecule_captioning": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "text_guided_molecule_generation": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "molecule_question_answering": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    "protein_question_answering": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
    },
    # "molecule_property_prediction": {
    #     "graphmvp": GraphMVP,
    # },
    # "pocket_molecule_docking": {
    #     "pharmolix_fm": PharmolixFM,
    # },
    # "structure_based_drug_design": {
    #     "pharmolix_fm": PharmolixFM,
    # },
    # "mutation_explanation": {
    #     "mutaplm": MutaPLM,
    # },
    # "mutation_engineering": {
    #     "mutaplm": MutaPLM,
    # }
}