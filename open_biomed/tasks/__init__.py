from typing import Optional

from open_biomed.tasks.multi_modal_tasks.text_based_molecule_editing import TextMoleculeEditing
from open_biomed.tasks.multi_modal_tasks.molecule_text_translation import MoleculeCaptioning
from open_biomed.tasks.multi_modal_tasks.text_guided_molecule_generation import TextGuidedMoleculeGeneration
from open_biomed.tasks.multi_modal_tasks.molecule_question_answering import MoleculeQA
from open_biomed.tasks.multi_modal_tasks.protein_question_answering import ProteinQA
from open_biomed.tasks.multi_modal_tasks.mutation_text_translation import MutationExplanation, MutationEngineering
from open_biomed.tasks.aidd_tasks.molecule_property_prediction import MoleculePropertyPrediction
from open_biomed.tasks.aidd_tasks.protein_molecule_docking import PocketMoleculeDocking
from open_biomed.tasks.aidd_tasks.structure_based_drug_design import StructureBasedDrugDesign

TASK_REGISTRY = {
    "text_based_molecule_editing": TextMoleculeEditing,
    "molecule_captioning": MoleculeCaptioning,
    "text_guided_molecule_generation": TextGuidedMoleculeGeneration,
    "molecule_question_answering": MoleculeQA,
    "protein_question_answering": ProteinQA,
    "molecule_property_prediction": MoleculePropertyPrediction,
    "pocket_molecule_docking": PocketMoleculeDocking,
    "structure_based_drug_design": StructureBasedDrugDesign,
    "mutation_explanation": MutationExplanation,
    "mutation_engineering": MutationEngineering,
}

def check_compatible(task_name: str, dataset_name: Optional[str], model_name: Optional[str]) -> None:
    # Check if the dataset and model supports the task
    # If not, raise NotImplementedError
    pass