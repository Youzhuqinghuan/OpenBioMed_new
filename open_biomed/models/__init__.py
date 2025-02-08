from open_biomed.models.foundation_models.molt5 import MolT5
from open_biomed.models.foundation_models.biot5 import BioT5
from open_biomed.models.foundation_models.biot5_plus import BioT5_PLUS
from open_biomed.models.foundation_models.graphmvp import GraphMVP
from open_biomed.models.foundation_models.pharmolix_fm import PharmolixFM

MODEL_REGISTRY = {
    "text_based_molecule_editing": {
        "molt5": MolT5,
        "biot5": BioT5,
        "biot5_plus": BioT5_PLUS
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
    "molecule_property_prediction": {
        "graphmvp": GraphMVP,
    },
    "structure_based_drug_design": {
        "pharmolix_fm": PharmolixFM,
    }
}