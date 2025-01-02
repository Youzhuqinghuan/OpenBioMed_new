from open_biomed.models.foundation_models.molt5 import MolT5
from open_biomed.models.foundation_models.biot5 import BioT5
from open_biomed.models.foundation_models.biot5_plus import BioT5_PLUS

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
}