from open_biomed.datasets.text_based_molecule_editing_dataset import FSMolEdit
from open_biomed.datasets.molecule_captioning_dataset import CheBI20ForMol2Text
from open_biomed.datasets.text_guided_molecule_generation_dataset import CheBI20ForText2Mol
from open_biomed.datasets.molecule_question_answering import MolQA
from open_biomed.datasets.protein_question_answering import ProteinQA

from open_biomed.datasets.molecule_property_prediction_dataset import MoleculeNet

DATASET_REGISTRY = {
    "text_based_molecule_editing":
        {
            "fs_mol_edit": FSMolEdit,  
            "fs_mol_multi": FSMolEdit 
        },
    "molecule_captioning":
        {
            "CheBI_20": CheBI20ForMol2Text
        },
    "text_guided_molecule_generation":
        {
            "CheBI_20": CheBI20ForText2Mol
        },
    "molecule_question_answering":
        {
            "MolQA": MolQA
        },
    "protein_question_answering":
        {
            "ProteinQA": ProteinQA
        },
    "molecule_property_prediction":
        {
            "BBBP": MoleculeNet,
            "Tox21": MoleculeNet,
            "ClinTox": MoleculeNet,
            "HIV": MoleculeNet,
            "Bace": MoleculeNet,
            "SIDER": MoleculeNet,
            "MUV": MoleculeNet,
            "Toxcast": MoleculeNet,
            # TODO: 以下是回归任务
            "FreeSolv": MoleculeNet,
            "ESOL": MoleculeNet,
            "Lipo": MoleculeNet,
            "qm7": MoleculeNet,
            "qm8": MoleculeNet,
            "qm9": MoleculeNet
        }
}