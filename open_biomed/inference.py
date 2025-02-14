import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from open_biomed.core.pipeline import InferencePipeline
from open_biomed.core.visualize import MoleculeVisualizer, ProteinVisualizer, ComplexVisualizer
from open_biomed.data import Molecule, Text, Protein, Pocket

def test_text_based_molecule_editing():
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/logs/text_based_molecule_editing/molt5-fs_mol_edit/train/checkpoints/last.ckpt",
        output_prompt="Edited molecule: {output}",
        device="cuda:0"
    )
    input_smiles = "Nc1[nH]c(C(=O)c2ccccc2)c(-c2ccccn2)c1C(=O)c1c[nH]c2ccc(Br)cc12"
    input_text = "This molecule can bind with recombinant human 15-LOX-1"
    outputs = pipeline.run(
        molecule=Molecule.from_smiles(input_smiles),
        text=Text.from_str(input_text),
    )
    print(f"Input SMILES: {input_smiles}")
    print(f"Input Text: {input_text}")
    print(outputs[0])
    return pipeline

def test_pocket_molecule_docking():
    pipeline = InferencePipeline(
        task="pocket_molecule_docking",
        model="pharmolix_fm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/pharmolix_fm.ckpt",
        output_prompt="Designed molecule: {output}",
        device="cuda:0"
    )
    protein = Protein.from_pdb_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/sbdd/4xli_B.pdb")
    ligand = Molecule.from_sdf_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/sbdd/4xli_B_ref.sdf")
    pocket = Pocket.from_protein_ref_ligand(protein, ligand)
    outputs = pipeline.run(
        molecule=ligand,
        pocket=pocket,
    )
    print(outputs)

def test_structure_based_drug_design():
    os.system("rm ./tmp/*.pkl")
    pipeline = InferencePipeline(
        task="structure_based_drug_design",
        model="pharmolix_fm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/pharmolix_fm.ckpt",
        output_prompt="Designed molecule: {output}",
        device="cuda:0"
    )
    protein = Protein.from_pdb_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/sbdd/4xli_B.pdb")
    ligand = Molecule.from_sdf_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/sbdd/4xli_B_ref.sdf")
    pocket = Pocket.from_protein_ref_ligand(protein, ligand)
    outputs = pipeline.run(
        pocket=pocket
    )
    print(outputs)
    return pipeline


def test_molecule_question_answering():
    pipeline = InferencePipeline(
        task="molecule_question_answering",
        model="molt5",
        model_ckpt="/AIRvePFS/dair/wenluo/projects/OpenBioMed_new/logs/molecule_question_answering/molt5-MolQA/train/checkpoints/last.ckpt",
        output_prompt="MQA: {output}",
        device="cuda:0"
    )
    input_text="COC(=O)C1=C2Nc3ccccc3[C@@]23CC[NH+]2CC=C[C@@]([C@@H](C)O)(C1)[C@H]23, Please identify if this molecule has a role as a conjugate acid, and if so, what is its paired conjugate base?"
    outputs = pipeline.run(
        text=Text.from_str(input_text)
    )
    print(outputs)
    return pipeline

def test_mutation_explanation():
    pipeline = InferencePipeline(
        task="mutation_explanation",
        model="mutaplm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/mutaplm.pth",
        output_prompt="Mutation effect: {output}",
        device="cuda:0"
    )
    # protein = Protein.from_fasta("MQPWHGKAMQRASEAGATAPKASARNARGAPMDPTESPAAPEAALPKAGKFGPARKSGSRQKKSAPDTQERPPVRATGARAKKAPQRAQDTQPSDATSAPGAEGLEPPAAREPALSRAGSCRQRGARCSTKPRPPPGPWDVPSPGLPVSAPILVRRDAAPGASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDIKDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAKEGNGFQEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFDEF")
    # mutation = "D95A"
    protein = Protein.from_fasta("MTLENVLEAARHLHQTLPALSEFGNWPTDLTATGLQPRAIPATPLVQALDQPGSPRTTGLVQAIRSAAHLAHWKRTYTEAEVGADFRNRYGYFELFGPTGHFHSTQLRGYVAYWGAGLDYDWHSHQAEELYLTLAGGAVFKVDGERAFVGAEGTRLHASWQSAAMSTGDQPILTFVLWRGEGLNALPRMDAA")
    mutation = "H163A"
    function = Text.from_str("Able to cleave dimethylsulfonioproprionate (DMSP) in vitro, releasing dimethyl sulfide (DMS). DMS is the principal form by which sulfur is transported from oceans to the atmosphere. The real activity of the protein is however subject to debate and it is unclear whether it constitutes a real dimethylsulfonioproprionate lyase in vivo: the very low activity with DMSP as substrate suggests that DMSP is not its native substrate.")
    outputs = pipeline.run(
        wild_type=protein,
        mutation=mutation,
        #function=function,
    )
    print(outputs)
    return pipeline

def test_mutation_engineering():
    pipeline = InferencePipeline(
        task="mutation_engineering",
        model="mutaplm",
        model_ckpt="/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/checkpoints/demo/mutaplm.pth",
        output_prompt="Designed mutation: {output}",
        device="cuda:1"
    )
    protein = Protein.from_fasta("MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK")
    prompt = Text.from_str("Strongly enhanced InsP6 kinase activity. The mutation in the ITPK protein causes a change in its catalytic activity.")
    outputs = pipeline.run(
        wild_type=protein,
        prompt=prompt
    )
    print(outputs)

def visualize_molecule():
    os.system("rm ./tmp/*.png")
    os.system("rm ./tmp/*.gif")
    pipeline = MoleculeVisualizer()
    ligand = Molecule.from_binary_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/mol_1739255667164_0.pkl")
    outputs = pipeline.run(
        ligand, mode="2D", rotate=False
    )
    print(outputs)
    return pipeline

def visualize_complex():
    os.system("rm ./tmp/*.png")
    os.system("rm ./tmp/*.gif")
    pipeline = ComplexVisualizer()
    ligand = Molecule.from_binary_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/mol_1739255667164_0.pkl")
    protein = Protein.from_pdb_file("/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_arch/tmp/sbdd/4xli_B.pdb")
    outputs = pipeline.run(molecule=ligand, protein=protein, rotate=True)
    print(outputs)
    return pipeline

INFERENCE_UNIT_TESTS = {
    "text_based_molecule_editing": test_text_based_molecule_editing,
    "pocket_molecule_docking": test_pocket_molecule_docking,
    "structure_based_drug_design": test_structure_based_drug_design,
    "visualize_molecule": visualize_molecule,
    "visualize_complex": visualize_complex,
    "molecule_question_answering": test_molecule_question_answering,
    "mutation_explanation": test_mutation_explanation,
    "mutation_engineering": test_mutation_engineering,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="visualize_complex")
    args = parser.parse_args()

    if args.task not in INFERENCE_UNIT_TESTS:
        raise NotImplementedError(f"{args.task} is not currently supported!")
    else:
        INFERENCE_UNIT_TESTS[args.task]()
    