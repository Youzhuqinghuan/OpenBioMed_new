from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# import function
from open_biomed.inference import test_text_based_molecule_editing, test_structure_based_drug_design, test_molecule_question_answering, visualize_molecule, visualize_complex
from open_biomed.data import Molecule, Text, Protein, Pocket

app = FastAPI()


# Define the request body model
class TaskRequest(BaseModel):
    task: str
    model: str
    input: dict


# Create a dictionary to store the pipeline instance
text_based_molecule_editing_pipeline = test_text_based_molecule_editing()
structure_based_drug_design_pipeline = test_structure_based_drug_design()
molecule_question_answering_pipeline = test_molecule_question_answering()
visualize_molecule_pipeline = visualize_molecule()
visualize_complex_pipeline = visualize_complex()

pipelines = {
    "text_based_molecule_editing": text_based_molecule_editing_pipeline,
    "structure_based_drug_design": structure_based_drug_design_pipeline,
    "molecule_question_answering": molecule_question_answering_pipeline,
    "visualize_molecule": visualize_molecule_pipeline,
    "visualize_complex": visualize_complex_pipeline
}


@app.post("/run_pipeline/")
async def run_pipeline(request: TaskRequest):
    task_name = request.task.lower()
    model = request.model.lower()
    input_data = request.input

    # Call the corresponding function based on the task name and get the pipeline instance
    if task_name == "text_based_molecule_editing":
        pipeline = pipelines["text_based_molecule_editing"]
        required_inputs = ["input_smiles", "input_text"]
        if not all(key in input_data for key in required_inputs):
            raise HTTPException(
                status_code=400, detail="input_smiles and input_text are required for text_based_molecule_editing task")
        outputs = pipeline.run(
            molecule=Molecule.from_smiles(input_data["input_smiles"]),
            text=Text.from_str(input_data["input_text"]),
        )
    elif task_name == "structure_based_drug_design":
        pipeline = pipelines["structure_based_drug_design"]
        required_inputs = ["protein_pdb_path", "ligand_sdf_path"]
        if not all(key in input_data for key in required_inputs):
            raise HTTPException(
                status_code=400, detail="protein_pdb_path and ligand_sdf_path are required for structure_based_drug_design task")
        protein = Protein.from_pdb_file(input_data["protein_pdb_path"])
        ligand = Molecule.from_sdf_file(input_data["ligand_sdf_path"])
        pocket = Pocket.from_protein_ref_ligand(protein, ligand)
        outputs = pipeline.run(
            pocket=pocket
        )
    elif task_name == "molecule_question_answering":
        pipeline = pipelines["molecule_question_answering"]
        required_inputs = ["input_text"]
        if not all(key in input_data for key in required_inputs):
            raise HTTPException(
                status_code=400, detail="input_text is required for molecule_question_answering task")
        outputs = pipeline.run(
            text=Text.from_str(input_data["input_text"])
        )
    elif task_name == "visualize_molecule":
        pipeline = pipelines["visualize_molecule"]
        required_inputs = ["ligand_pkl_path"]
        if not all(key in input_data for key in required_inputs):
            raise HTTPException(
                status_code=400, detail="ligand_pkl_path are required for visualize_molecule task")
        ligand = Molecule.from_binary_file(input_data["ligand_pkl_path"])
        outputs = pipeline.run(ligand, mode="2D", rotate=False)
    elif task_name == "visualize_complex":
        pipeline = pipelines["visualize_complex"]
        required_inputs = ["protein_pdb_path", "ligand_pkl_path"]
        if not all(key in input_data for key in required_inputs):
            raise HTTPException(
                status_code=400, detail="protein_pdb_path and ligand_pkl_path are required for visualize_complex task")
        ligand = Molecule.from_binary_file(input_data["ligand_pkl_path"])
        protein = Protein.from_pdb_file(input_data["protein_pdb_path"])
        outputs = pipeline.run(molecule=ligand, protein=protein, rotate=True)
    else:
        raise HTTPException(status_code=400, detail="Invalid task_name")

    return {"task": task_name, "model": model, "outputs": outputs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
